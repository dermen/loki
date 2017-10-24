import h5py
import numpy as np

f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/PE_cluster_difCor/GDP_buf/run9_cor.h5','r')
f_out = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/PE_cluster_difCor/GDP_buf/run9_all_diffcorr.h5','w')

keys = [kk for kk in f.keys() if kk.startswith('autocorr')]

all_diff_cor = []
for kk in keys:
    all_diff_cor.append(f[kk].value)

all_diff_cor = np.concatenate(all_diff_cor)

f_out.create_dataset('diff_corr', data = all_diff_cor)
f_out.close()
f.close()