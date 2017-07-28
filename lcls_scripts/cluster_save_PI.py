import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma
from sklearn.decomposition import PCA

import argparse
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compute difference correlation by consecutive pairing.')
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

parser.add_argument('-o','--out_dir', type=str,default = None,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-z','--zero_sigma', type=float,default = 2.0,
                   help='masking ceriterion: pixels within zero_sigma standard dev of zero are masked')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/',
                   help='where to look for the polar data')


parser.add_argument('-s','--start_ind', type=int, default = 0,
                   help='pixel index to start with, \
                   should more or less match the start and end indices of the interpolation')
parser.add_argument('-e','--end_ind', type=int, default = 350,
                   help='pixel index to start with')

parser.add_argument('-n','--num_clus', type=int,default = 10,
                   help='number of clusters for the 1st PC')


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


args = parser.parse_args()


# in this script, I will cluster by 1st PC of the radial profile and then pair by 2nd PC
run_num = args.run

num_clusters = args.num_clus

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir
if args.out_dir is None:
    save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/dif_cor/%s'%sample
else:
    save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

run_file = "run%d.tbl"%run_num


f_run = h5py.File(os.path.join(data_dir, run_file), 'r')

# load polar intensity and radial profiles
PI = f_run['polar_imgs']
RP = f_run['radial_profs']

num_q = PI.shape[1]


# Fileter by internsity first
s_ind = args.start_ind
e_ind  = args.end_ind
rp_protein = f_run['radial_profs'].value[:,s_ind:e_ind]

exclude = list(np.where(rp_protein.mean(-1)<10)[0])

#tags are the shots we are going to use after intensity filtering
tags = np.array( [i for i in range(rp_protein.shape[0]) if i not in exclude] )
rp_protein =rp_protein[tags]

#do the PCA
pca = PCA(n_components=2)
new_rp_protein = pca.fit_transform(rp_protein)

# histogram on the first PC
if num_clusters == 0:
    print("automatically chooseing numbers of cluster")
    hist = np.histogram(new_rp_protein[:,0], bins = 'fd')
else:
    print("binning into %d clusters"%num_clusters)
    hist = np.histogram(new_rp_protein[:,0], bins = num_clusters)
labels = np.digitize(new_rp_protein[:,0],bins=hist[1],right=True)

unique_labels = set(labels)
print("Number of clusters: %d"%len(unique_labels))
print("Total number of shots: %d"%len(labels))



# compute beam parameters, these are dummy values at the moment
k_beam = 0.0 
qvalues = np.linspace(0.1,1.5,3)

# for each cluster, randomly pair and compute difference int
cluster_sizes = []
corrs = []
unique_labels = set(labels)


# save PI
out_file = run_file.replace('.tbl','_clustered_PI.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')



for ll in unique_labels:
    print("consolidating cluster %d"%ll)
    class_member_mask = (labels == ll)
    shots_to_grab = tags[class_member_mask]

    if shots_to_grab.size>1:
        if shots_to_grab.size%2>0:
            shots_to_grab = shots_to_grab[:-1]
    else:
        continue
    
    shots_to_grab = sorted(shots_to_grab)
    
    
    shots = PI[shots_to_grab]
    # mask and normalize the shots
    if shots.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        shots = shots.astype(np.float64)
    
 
    for idx, ss in enumerate(shots):
        mask = make_mask(ss,zero_sigma=args.zero_sigma)
        ss *=mask
        mean_ss = ss.sum(-1)/mask.sum(-1) 

        ss = ss-mean_ss[:,None]
        shots[idx] = np.nan_to_num(ss*mask)

    f_out.create_dataset('PI_%d'%ll,data = shots)



f_out.create_dataset('mask_zero_sigma',data = args.zero_sigma)
f_out.close()

f_run.close()
