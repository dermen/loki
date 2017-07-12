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
save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/difInt_debug/%s'%sample
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


# open save file
out_file = run_file.replace('.tbl','_difInt.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')


# for each cluster, randomly pair and compute difference int

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
    
    shots = PI[shots_to_grab]
    # mask and normalize the shots
    
    if shots.dtype != 'float64':
        shots = shots.astype(np.float64)
 
    for idx, ss in enumerate(shots):
        mask = make_mask(ss,zero_sigma=1.5)

        ss *=mask
        mean_ss = ss.sum(-1)/mask.sum(-1) 

        ss = ss-mean_ss[:,None]
        shots[idx] = ss*mask

    f_out.create_dataset('difInt_%d'%ll, data = shots)

    dc = DiffCorr(shots, qvalues, 
        k_beam, pre_dif = False)
    corr = dc.autocorr()

    f_out.create_dataset('difCor_%d'%ll, data = corr)

f_out.close()

f_cluster.close()
f_run.close()
