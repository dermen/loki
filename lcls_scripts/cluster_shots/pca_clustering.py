import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

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

parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q used for pairing or the only q used for pairing')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q used for pairing or None')

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q',
                   help='where to look for the polar data')

parser.add_argument('-p','--num_pca', type=int, default=20,
                   help='number of pca components to use for clustering')


parser.add_argument('-c','--num_clusters', type=int, default=20,
                   help='number of clusters for kmeans')

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


args = parser.parse_args()


# in this script, I will cluster by correlations from single intensities
# the logic is that correlation curves from single intensities will exibihit mostly artifacts

run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir
cluster_type='pca%d_kmeans%d'%(args.num_pca, args.num_clusters)

save_dir = os.path.join(args.out_dir, cluster_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir = os.path.join( args.out_dir, cluster_type, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir
run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_PCA-cluster.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')


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
    print('PCA clustering for qidx %d'%qidx)
    f_out.create_group('q%d'%qidx)
    shots=PI[:,qidx,:][:,None,:]
    this_mask = mask[qidx][None,:]

    norm_shots = np.zeros_like(shots)

    for idx,ss in enumerate(shots):
        norm_shots[idx]=normalize_shot(ss,this_mask)
    
    pca=PCA(n_components=args.num_pca)
    new_norm_shots=pca.fit_transform(norm_shots[:,0,:])

    kmeans=KMeans(n_clusters=args.num_clusters)
    kmeans.fit(new_norm_shots)

    f_out.create_dataset('q%d/cluster_labels'%qidx,data=kmeans.labels_)
    f_out.create_dataset('q%d/explained_variance_ratio'%qidx,data=pca.explained_variance_ratio_)
    f_out.create_dataset('q%d/kmeans_num_clusters'%qidx,data=args.num_clusters)

    del shots
    del norm_shots
print ("done!")
f_out.close()