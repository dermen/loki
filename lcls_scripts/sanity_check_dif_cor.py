import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma

import argparse
import numpy as np

import matplotlib.pyplot as plt

# load the water run
f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/test_simulated_data/\
compact_001_withLigs_downsamp_q0.18_0.90_Nq35.hdf5','r')
f_out = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/test_simulated_data/\
compact_001_withLigs_downsamp_q0.18_0.90_Nq35_cors.h5','w')
#######################
use_basic_mask= True
pmask_basic = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
#######################
meanNorm = False

PI = f['downsamp_shots'].value[:,1:,:]
assert(pmask_basic.shape[0]==PI.shape[1])
assert(pmask_basic.shape[-1]==PI.shape[-1])

cluster_shot_tags = range(PI.shape[0])

if len(cluster_shot_tags)<2:
    print "skipping little cluster %d in big cluster %d"%(cc,cluster_num)


shots = PI[cluster_shot_tags]

# mask and normalize the shots
if shots.dtype != 'float64':
    # shots need to be float64 or more. 
    # float32 resulted in quite a bit of numerical error 
    shots = shots.astype(np.float64)

norm_shots = np.zeros_like(shots)
masked_shots = np.zeros_like(shots)
meanNorm_shots=np.zeros_like(shots)

for idx, ss in enumerate(shots):
    if use_basic_mask:
        mask = pmask_basic
    else:
        mask = make_mask(ss,zero_sigma=0.0)
    ss *=mask
    masked_shots[idx] = ss

    mean_ss = ss.sum(-1)/mask.sum(-1) 

    ss = ss-mean_ss[:,None]
    
    meanNorm_shots [idx] = np.nan_to_num(ss/mean_ss[:,None]*mask)
    norm_shots[idx] = np.nan_to_num(ss*mask)

#clean up a bit
del shots

diff_norm = norm_shots[1::2]-norm_shots[::2]
diff_masked = masked_shots[1::2]-masked_shots[::2]
diff_meanNorm = meanNorm_shots[1::2]-meanNorm_shots[::2]

# save difference int
# f_out.create_dataset('norm_diff', data = diff_norm)
# f_out.create_dataset('masked_diff', data = diff_masked)

qs = np.linspace(0.2,0.9,PI.shape[1])
dc = DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_ac=dc.autocorr().mean(0)

dc = DiffCorr(diff_norm,qs,0,pre_dif=True)
ac = dc.autocorr().mean(0)/mask_ac
f_out.create_dataset('norm_corr',data=ac)

dc = DiffCorr(diff_masked,qs,0,pre_dif=True)
ac = dc.autocorr().mean(0)/mask_ac
f_out.create_dataset('masked_corr',data=ac)

dc = DiffCorr(diff_meanNorm,qs,0,pre_dif=True)
ac = dc.autocorr().mean(0)/mask_ac
f_out.create_dataset('meanNorm_corr',data=ac)

f_out.close()



