import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma

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

parser.add_argument('-d','--out_dir', type=str,default = None,
                   help='output dir to save in, overwrites the sample type dir')

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

run_num = args.run
if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file and pairing file

data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/'
cluster_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/'
if args.out_dir is None:
    save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/dif_cor/%s'%sample
else:
    save_dir = args.out_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

run_file = "run%d.tbl"%run_num
cluster_file = "run%d_cluster.h5"%run_num

f_run = h5py.File(os.path.join(data_dir, run_file), 'r')
f_cluster = h5py.File(os.path.join(cluster_dir, cluster_file), 'r')

# load tags and labels
tags = f_cluster['shot_inds'].value
labels = f_cluster['cluster_labels'].value
# load polar intensity
PI = f_run['polar_imgs']

# compute beam parameters, these are dummy values at the moment
k_beam = 0.0 
qvalues = np.linspace(0.1,1.5,9)

# for each cluster, randomly pair and compute difference int
cluster_sizes = []
corrs = []
unique_labels = set(labels)
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
    cluster_sizes.append(len(shots_to_grab))
    
    shots = PI[shots_to_grab]
    # mask and normalize the shots
   
    for idx, ss in enumerate(shots):
        mask = make_mask(ss)

        ss *=mask
        mean_ss = ss.sum(-1)/mask.sum(-1) 

        ss = ss-mean_ss[:,None]
        shots[idx] = ss*mask


    dc = DiffCorr(shots, qvalues, 
        k_beam, pre_dif = False)
    corr = dc.autocorr().mean(0)

    corrs.append(corr)


total_shots = np.sum(cluster_sizes).astype(float)
print ("total number of shots used is %d"%total_shots)

# diff cor for the whole run
cluster_sizes = np.array(cluster_sizes)/total_shots
corrs = np.array(corrs)
ave_corr = (corrs * cluster_sizes[:,None,None]).sum(0)

# save ave diff cor
out_file = run_file.replace('.tbl','_cor.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')
f_out.create_dataset('ave_cor',data = ave_corr)
f_out.create_dataset('num_shots',data = total_shots)
f_out.close()

f_cluster.close()
f_run.close()
