import h5py
import numpy as np
from loki.RingData import DiffCorr

f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/PE_cluster_difCor/h2o/water_diffcorr.tbl','r')
diff_corr = f['diff_corr']

f_out = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/PE_cluster_difCor/h2o/water_sigConverge.tbl','w')


#get mask 
mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
qs = np.linspace(0.1,1.0, diff_corr.shape[1])
mask_dc = DiffCorr(mask[None,:], qs, 0, pre_dif=True)
mask_corr = mask_dc.autocorr().mean(0)

num_samples = 200
results = np.zeros( (num_samples, diff_corr.shape[1], diff_corr.shape[2]) )
shot_inds = np.arange(diff_corr.shape[0])

num_shots = 30000
for nn in range(num_samples):
    np.random.shuffle(shot_inds)

    idx = sorted(shot_inds[:num_shots])

    results[nn] = diff_corr[idx,:,:].mean(0) / mask_corr
f_out.create_dataset('diff_corrs',data=results)
