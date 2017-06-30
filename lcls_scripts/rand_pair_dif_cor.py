import h5py
from loki.RingData import DiffCor


import argparse

parser = argparse.ArgumentParser(description='Cluster shots in the same run \n\
    by running PCS on radial profs and using hierarchical clustering.')
parser.add_argument('-r','--run', type=int,
                   help='run number')


args = parser.parse_args()

run_num = args.run

# import run file and pairing file

data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/'
cluster_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/'
run_file = "run%d.tbl"%run_num
cluster_file = "run%d_cluster.h5"%run_num

f_run = h5py.File(os.path.join(data_dir, run_file), 'r')
f_cluster = h5py.File(os.path.join(cluster_dir, cluster_file), 'r')

# load tags and labels
tags = f_cluster['shot_ind'].value
labels = f_cluster['cluster_labels'].value
# load polar intensity
PI = f_run['polar_imgs']


# for each cluster, randomly pair and compute difference int
cluster_sizes = []
unique_labels = set(labels)
for ll in unique_labels:

    class_member_mask = (labels == ll)
    shots_to_grab = tags[class_member_mask]
    

# diff cor for the whole run


# save diff int and ave diff cor