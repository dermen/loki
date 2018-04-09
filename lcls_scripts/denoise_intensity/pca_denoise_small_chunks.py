
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

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/results/combined_tables/finer_q',
                   help='where to look for the polar data')

parser.add_argument('-c','--chunk_size', type=int, default=1000,
                   help='size of the chunks of shots to run PCA on')

parser.add_argument('-m','--min_pca', type=int, default=3,
                   help='the minimum number of PCA components to remove from small chunks')




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
    

def get_elbow(components,partial_mask):
    masked_eigenimgs = reshape_unmasked_values_to_shots( components.astype(np.float64),
                                                    partial_mask)
    qs = np.linspace(0,1,1)
    dc = DiffCorr(masked_eigenimgs,q_values=qs,k=0)
    eigenimg_ac = dc.autocorr()[:,:,1]

    if np.abs(eigenimg_ac[0])<0.5:
        cutoff=1
    else:
        for ii,aa in enumerate(eigenimg_ac[:,0]):
            if np.abs(aa)>=0.5:
                continue
            else:
                cutoff=ii
                break
    return cutoff

args = parser.parse_args()



run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir
save_dir = args.out_dir


save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir


run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_PCA-smallChunks-denoise.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'a')

# if 'polar_mask_binned' in f.keys():
#     mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
# else:
#ever run uses this mask now!
mask = np.load('/reg/d/psdm/cxi/cxilp6715/results/shared_files/binned_pmask_basic4.npy')


PI = f['polar_imgs']
shot_tags = np.arange(0,PI.shape[0])

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
    else:
        print('has data for %s'%q_group)
        continue

    shots=PI[:,qidx,:][:,None,:]
    this_mask = mask[qidx][None,:]
    partial_mask=this_mask.copy()
    print('normaling shots...')
    norm_shots = np.zeros_like(shots)
    for idx,ss in enumerate(shots):
        norm_shots[idx]=normalize_shot(ss,this_mask)

    # now I shall do pca denoise by smaller chunks
    num_shots = norm_shots.shape[0]
    chunk_size = args.chunk_size
    num_chunks = int(num_shots/chunk_size)
    all_corrs=[]
    all_nums=[]
    pca_cutoffs=[]
    for n_chunk in range(num_chunks):
        Train = norm_shots[n_chunk*chunk_size:(n_chunk+1)*chunk_size, partial_mask==1] 
        if Train.shape[0]%2>0:
            Train=Train[:-1]
        # print Train.shape
        #run pca, and find elbow
        pca=PCA(n_components=20, whiten = False) 
        new_Train=pca.fit_transform(Train)
        components=pca.components_
        num_pca= get_elbow(components, partial_mask)

        assert(num_pca>0)
        # remove at least two PCs
        if num_pca<args.min_pca:
            num_pca=args.min_pca
            
        pca_cutoffs.append(num_pca)
        qvalues = np.linspace(0,1,partial_mask.shape[0])
        print('Chunk %d: denoisng with PCA critical num_pca_components = %d...'%(n_chunk,num_pca) )
    
        # get back the masked images and components
 
        masked_mean_train =reshape_unmasked_values_to_shots(Train,partial_mask).mean(0)

        # denoise
  
        Train_noise = new_Train[:,:num_pca].dot(components[:num_pca])
        denoise_Train= reshape_unmasked_values_to_shots(Train-Train_noise-Train.mean(0)[None,:]
                                                    , partial_mask)
        
        if denoise_Train.shape[0]%2>0:
            denoise_Train=denoise_Train[:-1]
        denoise_Train=denoise_Train[:-1][::2]-denoise_Train[1:][::2]
        dc=DiffCorr(denoise_Train,qvalues,0,pre_dif=True)
        Train_difcor= dc.autocorr().mean(0)[0]
        
        all_corrs.append(Train_difcor)
        all_nums.append(Train.shape[0])

        f_out.create_dataset('q%d/dif_cor%d'%(qidx, n_chunk)
            ,data=Train_difcor)


    all_corrs = np.array(all_corrs)
    all_nums = np.array(all_nums)
    # print all_corrs.shape
    # print all_nums

    ave_corr = np.sum(all_corrs*(all_nums/float(all_nums.sum()))[:,None],axis=0 )
    # print ave_corr.shape
    f_out.create_dataset('q%d/ave_dif_cor'%qidx,data=ave_corr)
    f_out.create_dataset('q%d/chunk_sizes'%qidx,data=all_nums)
    f_out.create_dataset('q%d/pca_cutoffs'%qidx,data=np.array(pca_cutoffs))
    
    
    if 'num_shots' not in f_out[q_group].keys():        
        f_out.create_dataset('q%d/num_shots'%qidx, data=all_nums.sum())
    del shots
    del norm_shots
    del Train

print ("done!")
f_out.close()