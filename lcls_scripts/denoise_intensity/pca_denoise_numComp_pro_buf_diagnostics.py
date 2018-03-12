
# take a run, do PCA at each q

# save the shots into train and test, save the 
import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

import sys

from loki.RingData.DiffCorr import DiffCorr


parser = argparse.ArgumentParser(description='Compute difference correlation by pairing single intensity correlations.')
parser.add_argument('-r','--run', type=int,
                   help='run number')
parser.add_argument('-t','--samp_type', type=int,
                   help='type of data/n \
# Sample IDs\n\
# -1: Silver Behenate smaller angle\n\
# -2: Silver Behenate wider angle\n\
# 0: GDP buffer\n\
# 1: ALF BUffer\n\
# 2: GDP protein\n\
# 3: ALF protein\n\
# 4: Water \n\
# 5: Helium\n\
# 6: 3-to-1 Recovered GDP')

parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q used for pairing or the only q used for pairing')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q used for pairing or None')

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q',
                   help='where to look for the polar data')



parser.add_argument('-p','--num_pca', type=int, default=None,
                   help='num_pca+1 is the max number of pca components to subtract')




def sample_type(x):
    return {-1:'AgB_sml',
    -2:'AgB_wid',
     0:'GDP_buf',
     1:'ALF_buf',
     2:'GDP_pro',
     3:'ALF_pro',
     4:'h2o',
     5:'he',
     6:'3to1_rec_GDP_pro'}[x]

def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)

def reshape_unmasked_values_to_shots(shots,mask):
    # this takes vectors of unmasked values, and reshaped this into their masked forms
    # mask is 2D, shots are 1D
    assert(shots.shape[-1]==np.sum(mask) )
    flat_mask = mask.flatten()
    reshaped_shots = np.zeros( (shots.shape[0],mask.size), dtype=shots.dtype)
    
    reshaped_shots[:, flat_mask==1] = shots
    
    return reshaped_shots.reshape((shots.shape[0],mask.shape[0],mask.shape[1]))
    


args = parser.parse_args()



run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file




data_dir = args.data_dir


if sample in ['ALF_pro','GDP_pro']:
    if sample.startswith('ALF'):
        f_buf = h5py.File(os.path.join(data_dir,'run48.tbl'))
    else:
        f_buf = h5py.File(os.path.join(data_dir,'run13.tbl'))
else:
    print('need protein sample')
    sys.exit()
save_dir = args.out_dir

if args.num_pca is None:
    num_pca_file = os.path.join(save_dir,'num_pca_components.txt')

    if not os.path.exists(num_pca_file):
        print("there is no num_pca_components.txt file in %s"%save_dir)
        sys.exit()
    num_pca_components = np.loadtxt(num_pca_file)


save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir


run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_PCA-denoise.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'a')

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')


PI = f['polar_imgs']
PI_buf = f_buf['polar_imgs']

# figure which qs are used for pairing
qmin = args.qmin
qmax = args.qmax

if qmax is None:
    qcluster_inds = [qmin]
else:
    qcluster_inds = range(qmin,qmax+1) # qmax is included

# this script on does the clustering only

# normalize all the shots at each q index

for qidx in qcluster_inds:
    print('PCA denoising for qidx %d'%qidx)
    q_group = 'q%d'%qidx
    if q_group not in f_out.keys():
        f_out.create_group(q_group)
    shots=PI[:,qidx,:][:,None,:]
    num_pro_shots = shots.shape[0]
    buf_shots = PI_buf[:,qidx,:][:,None,:]
    
    #combine buffer and pro shots 
    shots = np.concatenate([shots,buf_shots])
    print('num of protein shots: %d'%num_pro_shots)
    print ('num of buffer shots: %d'%(shots.shape[0]-num_pro_shots))

    this_mask = mask[qidx][None,:]
    print('normaling shots...')
    norm_shots = np.zeros_like(shots)
    for idx,ss in enumerate(shots):
        norm_shots[idx]=normalize_shot(ss,this_mask)
    # do we want to normalize by the entire range of intensity?
    # divide into Train and test
    num_shots = norm_shots.shape[0]
    cutoff = int(num_pro_shots*0.1) # use 10% of the protein shots as testing set
    partial_mask = this_mask.copy()
    Train = norm_shots[cutoff:, partial_mask==1]
    Test = norm_shots[:cutoff, partial_mask==1]

    print ("%d test shots"%(Test.shape[0]))
    print ("%d train shots"%(Train.shape[0]))

    qvalues = np.linspace(0,1,partial_mask.shape[0])
    mask_dc = DiffCorr(partial_mask,qvalues,0, pre_dif=True)
    mask_cor = mask_dc.autocorr()
    if args.num_pca is None:
        num_pca = int(num_pca_components[qidx])
        max_pca = num_pca+5
    else:
        num_pca = args.num_pca+1
        max_pca = args.num_pca+1

    print('denoisng with PCA critical num_pca_components = %d...'%num_pca)
    if 'pca_components' not in f_out[q_group].keys():
        # if there is no pca component saved, then run it and save the components
        pca=PCA(n_components=50, whiten = False)

        new_Train=pca.fit_transform(Train)
        new_Test = pca.transform(Test)
        if 'explained_variance_ratio' not in f_out[q_group].keys():
            f_out.create_dataset('q%d/explained_variance_ratio'%qidx,data=pca.explained_variance_ratio_)
            f_out.create_dataset('q%d/new_train_pro'%qidx, 
                data = new_Train[:(num_pro_shots-cutoff),:])
            f_out.create_dataset('q%d/new_train_buf'%qidx,
            data = new_Train[(num_pro_shots-cutoff):])
            f_out.create_dataset('q%d/new_test_pro'%qidx, 
                data = new_Test)

        # get back the masked images and components
        components=pca.components_
        f_out.create_dataset('q%d/pca_components'%qidx,data=components)
    else:
        #load the components previously save the then do the transformations
        components = f_out['q%d/pca_components'%qidx].value
        _m = Train.astype(np.float64).mean(0)
        new_Train = (Train.astype(np.float64)-_m).dot(components.T)

        _m = Test.astype(np.float64).mean(0)
        new_Test = (Test.astype(np.float64)-_m).dot(components.T)

        if 'new_test_pro' not in f_out[q_group].keys():
            f_out.create_dataset('q%d/new_train_pro'%qidx, 
                data = new_Train[:(num_pro_shots-cutoff),:])
            f_out.create_dataset('q%d/new_train_buf'%qidx,
            data = new_Train[(num_pro_shots-cutoff):])
            f_out.create_dataset('q%d/new_test_pro'%qidx, 
                data = new_Test)
    # get back the masked images and components
       
    masked_mean_train =reshape_unmasked_values_to_shots(Train,partial_mask).mean(0)
    masked_mean_test =reshape_unmasked_values_to_shots(Test,partial_mask).mean(0)

    # denoise
    for nn in range(max_pca):
        pca_group = 'q%d/pca%d'%(qidx,nn)
        if 'pca%d'%nn not in f_out[q_group].keys():
            f_out.create_group(pca_group)
        else:
            print("pca denoise at pca n_components = %d is already done. Skip!"%nn)
            continue

        if nn>0:

            print('subtracting noise with pca n_components =  %d'%nn)
            Test_noise = new_Test[:,:nn].dot(components[:nn])
            denoise_Test = reshape_unmasked_values_to_shots(Test-Test_noise-Test.mean(0)[None,:],
            partial_mask)
            Train_noise = new_Train[:,:nn].dot(components[:nn])
            denoise_Train= reshape_unmasked_values_to_shots(Train-Train_noise-Train.mean(0)[None,:]
                                                        , partial_mask)

            denoise_Train_pro = denoise_Train[:(num_pro_shots-cutoff)]
            denoise_Train_buf = denoise_Train[(num_pro_shots-cutoff):]

            print('num of protein shots: %d'%denoise_Train_pro.shape[0])
            print ('num of buffer shots: %d'%(denoise_Train_buf.shape[0]))

            dc=DiffCorr(denoise_Train_pro,qvalues,0,pre_dif=False)
            Train_pro_difcor= (dc.autocorr()/mask_cor).mean(0)

            dc=DiffCorr(denoise_Train_buf,qvalues,0,pre_dif=False)
            Train_buf_difcor= (dc.autocorr()/mask_cor).mean(0)

            dc=DiffCorr(denoise_Test,qvalues,0,pre_dif=False)
            Test_difcor= (dc.autocorr()/mask_cor).mean(0)


            f_out.create_dataset('q%d/pca%d/test_pro_difcor'%(qidx,nn)
                ,data=Test_difcor)
            f_out.create_dataset('q%d/pca%d/train_pro_difcor'%(qidx,nn)
                ,data=Train_pro_difcor)
            f_out.create_dataset('q%d/pca%d/train_buf_difcor'%(qidx,nn)
                ,data=Train_buf_difcor)
    
        else:
            print('not doing denoising, just computing baseline')

            dc=DiffCorr(norm_shots[cutoff:num_pro_shots],qvalues,0,pre_dif=False)
            difcor= (dc.autocorr()/mask_cor).mean(0)
            f_out.create_dataset('q%d/pca%d/train_pro_difcor'%(qidx,nn)
            ,data=difcor)
            
            dc=DiffCorr(norm_shots[num_pro_shots:],qvalues,0,pre_dif=False)
            difcor= (dc.autocorr()/mask_cor).mean(0)
            f_out.create_dataset('q%d/pca%d/train_buf_difcor'%(qidx,nn)
            ,data=difcor)
            
            dc=DiffCorr(norm_shots[:cutoff],qvalues,0,pre_dif=False)
            difcor= (dc.autocorr()/mask_cor).mean(0)
            f_out.create_dataset('q%d/pca%d/test_pro_difcor'%(qidx,nn)
                ,data=difcor)
    
    if 'num_pro_shots' not in f_out[q_group].keys():        
        f_out.create_dataset('q%d/num_pro_shots'%qidx, data=num_pro_shots)
        f_out.create_dataset('q%d/num_buf_shots'%qidx, data=norm_shots.shape[0]-num_pro_shots)
    del shots
    del norm_shots

print ("done!")
f_out.close()