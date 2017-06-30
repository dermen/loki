import h5py
from loki.RingData import DiffCorr
import os

import numpy.ma as ma

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Cluster shots in the same run \n\
    by running PCS on radial profs and using hierarchical clustering.')
parser.add_argument('-r','--run', type=int,
                   help='run number')


args = parser.parse_args()

run_num = args.run

# import run file and pairing file

data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/'
cluster_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/'
save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/dif_cor/'
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
        mask_shot = ma.MaskedArray(ss, ~mask)
        mask_shot = 

        ss[mask] -= np.mean(ss[mask])


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
