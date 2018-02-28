import numpy as np
import h5py

from sklearn.cluster import KMeans

import os

from scipy.interpolate import interp1d

import argparse
import sys

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


parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')


parser.add_argument('-d','--data_dir', type=str, required=True,
                   help='where to look for the difference correlations')


parser.add_argument('-n','--diff_norm', type=str, required=True,
                   help='if yes, subtract the normalized versions of protein from buffer')

parser.add_argument('-c','--n_cluster', type=int, default=20,
                   help='number of clusters to use in the kmeans clustering of PCA cluster average')

parser.add_argument('-p','--phi_offset', type=int, default=15,
                   help='number of phi pixels to ignore')

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

def normalize(d, zero=True):
    x=d.copy()
    if zero:
        x-=x.min()
    return x/(x.max()-x.min())

def normalize_set(shot_set):
    norm_shot_set = np.zeros_like(shot_set)
    for iq in range(shot_set.shape[0]):
        norm_shot_set[iq] = normalize(shot_set[iq])
    return norm_shot_set

args = parser.parse_args()

if args.diff_norm in ['y','yes']:
    diff_norm=True
else:
    diff_norm=False

run_num = int(args.run)

# in this script, I will cluster by correlations from single intensities
# the logic is that correlation curves from single intensities will exibihit mostly artifacts

if args.samp_type not in [2,3]:
    print("Error!!!! cannot do buffer subtraction for this type of sample")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
    if sample=='GDP_pro':
        buffer_name = 'GDP_buf'
    if sample=='ALF_pro':
        buffer_name = 'ALF_buf'


save_dir = os.path.join( args.out_dir, "%s_sansBuffer"%sample)
data_dir = args.data_dir
pro_dir = os.path.join( args.data_dir, sample)
buf_dir = os.path.join( args.data_dir, buffer_name)

save_dir = os.path.join( args.out_dir, sample)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if diff_norm:
    out_file = 'run%d_norm_sansBuffer.h5'%run_num
else:
    out_file = 'run%d_sansBuffer.h5'%run_num
f_out = h5py.File(os.path.join(save_dir, out_file),'w')
print('saving results in %s'%out_file)


f_ALF = h5py.File(os.path.join(pro_dir,'run%d_cor.h5'%run_num),'r')
q_keys= f_ALF.keys()
q_inds= [int(kk.split('q')[-1]) for kk in q_keys ]


f_ALFbuf = h5py.File(os.path.join(buf_dir,'all_difcor.h5'),'r')

phi_offset = args.phi_offset
num_phi = f_ALF[q_keys[0]]['ave_norm_corr'].shape[-1]

for qidx in q_inds:
    f_out.create_group('q%d'%qidx)

    # gather protein cluster difcors
    cluster_aves=[]
    ALFpro_difcor_keys = [k for k in f_ALF['q%d'%(qidx)].keys() if k.endswith('difcors')]

    for kk in ALFpro_difcor_keys:
        x= f_ALF['q%d'%(qidx)][kk].value[:,phi_offset:num_phi/2-phi_offset]
        cluster_aves.append(x.mean(0))
        del x
    ALFpro_corrPair_cluster_ave =np.array(cluster_aves)
    
    # gather buffer cluster difcors
    ALFbuf_difcor_keys = [k for k in f_ALFbuf['q%d'%(qidx)].keys() if k.endswith('difcors')]
    cluster_aves=[]

    for kk in ALFbuf_difcor_keys:
        x =f_ALFbuf['q%d'%(qidx)][kk].value[:,phi_offset:num_phi/2-phi_offset]
        cluster_aves.append(x.mean(0))
        del x
        
    ALFbuf_corrPair_cluster_ave = np.array(cluster_aves)


    #normalize the cluster averages and then sort them using kmeans
    norm_ALFpro_cluster_ave = np.zeros_like(ALFpro_corrPair_cluster_ave)

    for iq in range(ALFpro_corrPair_cluster_ave.shape[0]):
        norm_ALFpro_cluster_ave[iq] = normalize(ALFpro_corrPair_cluster_ave[iq])
        
    norm_ALFbuf_cluster_ave = np.zeros_like(ALFbuf_corrPair_cluster_ave)
    for iq in range(ALFbuf_corrPair_cluster_ave.shape[0]):
        norm_ALFbuf_cluster_ave[iq] = normalize(ALFbuf_corrPair_cluster_ave[iq])
        
    cutoff = norm_ALFbuf_cluster_ave.shape[0]

    # cluster and sort
    n_cluster=args.n_cluster
    kmeans = KMeans(n_clusters=n_cluster)
    X = np.concatenate((norm_ALFbuf_cluster_ave,norm_ALFpro_cluster_ave))
    norm_labels=kmeans.fit_predict(X)

    cutoff = norm_ALFbuf_cluster_ave.shape[0]

    # what about normalized cors?
    # go through the clusters, gather all the proteins and all the buffers, pair them
    km_cluster_norm_diff_cor=[]
    cluster_shot_nums=[]
    pro_clusters_not_used = []
    for ll in np.unique(norm_labels):
        inds=np.where(norm_labels==ll)[0]
        if np.sum(inds>=cutoff)==0 or np.sum(inds<cutoff)==0:
            if np.sum(inds>=cutoff)>0 and np.sum(inds<cutoff)==0:
                #then some of the protein shots are not used
                #let's remember what clusters they are
                pro_inds=inds[inds>=cutoff]-cutoff
                print pro_inds
                keypro = np.array(ALFpro_difcor_keys)[pro_inds]
                print keypro
                pro_clusters_not_used.append(keypro )

            continue
        
        pro_inds=inds[inds>=cutoff]-cutoff
        buf_inds=inds[inds<cutoff]
        
        pro_difcors=[]
        for ii in pro_inds:
            keypro = ALFpro_difcor_keys[ii]
            pro_difcors.append( f_ALF['q%d'%(qidx)][keypro][:,phi_offset:num_phi/2-phi_offset])
            
        buf_difcors=[]
        for ii in buf_inds:
            keybuf = ALFbuf_difcor_keys[ii]
            buf_difcors.append( f_ALFbuf['q%d'%(qidx)][keybuf][:,phi_offset:num_phi/2-phi_offset])

        if diff_norm:
            pro_difcors=normalize_set(np.concatenate(pro_difcors))
            buf_difcors=normalize_set(np.concatenate(buf_difcors))
        else:
            pro_difcors=np.concatenate(pro_difcors)
            buf_difcors=np.concatenate(buf_difcors)
        
        optimal_d=np.zeros_like(pro_difcors)
        for ii in range(optimal_d.shape[0]):
            iy=np.mean((buf_difcors-pro_difcors[ii])**2,axis=-1).argmin()
            optimal_d[ii]=pro_difcors[ii]-buf_difcors[iy]

        km_cluster_norm_diff_cor.append(optimal_d)
        cluster_shot_nums.append(optimal_d.shape[0])

        
    km_cluster_norm_diff_cor=np.concatenate(km_cluster_norm_diff_cor)
    f_out.create_dataset('q%d/cluster_shot_nums'%qidx, data=np.array(cluster_shot_nums))
    f_out.create_dataset('q%d/diff_cor'%qidx, data = km_cluster_norm_diff_cor)
    if len(pro_clusters_not_used)>0:
        pro_clusters_not_used = np.concatenate(pro_clusters_not_used).astype(str)
        f_out.create_dataset('q%d/pro_clusters_not_used'%qidx, 
            data = pro_clusters_not_used)

f_out.close()