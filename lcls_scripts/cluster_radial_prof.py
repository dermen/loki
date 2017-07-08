import numpy as np
import h5py
import sys
import os

import matplotlib.pyplot as plt


from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


import argparse

parser = argparse.ArgumentParser(description='Cluster shots in the same run \n\
    by running PCS on radial profs and using hierarchical clustering.')
parser.add_argument('-r','--run', type=int,
                   help='run number')

parser.add_argument('-n','--num_clus', type=int,default = 10,
                   help='number of clusters')

parser.add_argument('-s','--num_shots', type=int,default = None,
                   help='number of shots to cluster')


parser.add_argument('-i','--start_ind', type=int, default = 0,
                   help='shot index to start with')

parser.add_argument('-m','--method', type=str, default = 'hist',
                   help='clustering method to use')


args = parser.parse_args()
run_num = args.run
start_ind = args.start_ind
num_clusters = args.num_clus
method = args.method

if args.num_shots is None:
    end_ind = None
else:
    end_ind = start_ind + args.num_shots
# load file

data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/'
save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/'

run_file = "run%d.tbl"%run_num
if end_ind is None:
    end_ind = int(1e6)


f = h5py.File(os.path.join(data_dir, run_file), 'r')


# load data and create tags
rp_protein = f['radial_profs'].value[:,:-100]
all_tags =  range(f['radial_profs'].shape[0])
# filter by average intensity
thresh = 10
exclude = list(np.where(rp_protein.mean(-1)<thresh)[0])
exclude.append(51519)


tags = [i for i in all_tags if i not in exclude]
rp_protein =rp_protein[tags]



# perform PCA
pca = PCA(n_components=2)
new_rp_protein = pca.fit_transform(rp_protein)


# cluster by PCA

data = new_rp_protein[start_ind:end_ind]

# histogram on the first PC
if method=='hist':
    if num_clusters == 0:
	hist = np.histogram(data[:,0], bins = 'fd')
    else:
        hist = np.histogram(data[:,0], bins = num_clusters)
    labels = np.digitize(data[:,0],bins=hist[1],right=True)
elif method == 'h_average':
    clustering = AgglomerativeClustering(linkage='average', n_clusters=num_clusters)
    clustering.fit(data)
    labels = clustering.labels_
else:
    print("ERROR!!! clustering method not available.")
    sys.exit()

unique_labels = set(labels)
print("Number of clusters: %d"%len(unique_labels))
print("Total number of shots: %d"%len(labels))

# save tags (which shots are use) and labels (which cluster shots belong to)
save_file = run_file.replace('.tbl','_cluster.h5')
f_out = h5py.File(os.path.join(save_dir,save_file), 'w')
f_out.create_dataset('shot_inds', data = np.array(tags)[start_ind:end_ind])
f_out.create_dataset('cluster_labels',data = labels)
f_out.close()
