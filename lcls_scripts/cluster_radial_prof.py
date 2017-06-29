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


args = parser.parse_args()
run_num = args.run
# load file

data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/'
save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/'
run_file = "run%d.tbl"%run_num
num_clusters = 10
start_ind = 1000
end_ind = 2000


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

clustering = AgglomerativeClustering(linkage='average', n_clusters=num_clusters)
clustering.fit(data)


core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[a.core_sample_indices_] = True

labels = clustering.labels_


# save tags (which shots are use) and labels (which cluster shots belong to)
save_file = run_file.replace('.tbl','_cluster.h5')
f_out = h5py.File(os.path.join(save_dir,save_file), 'w')
f_out.create_dataset('shot_inds', data = np.array(tags)[start_ind,end_ind])
f_out.create_dataset('cluster_labels',data = labels)
f_out.close()