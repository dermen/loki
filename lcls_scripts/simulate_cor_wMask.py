import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma

import argparse
import numpy as np
import glob

import matplotlib.pyplot as plt
import re


parser = argparse.ArgumentParser(description='Compute difference correlation of simulated polar intensities while applying a mask.')
parser.add_argument('-d','--dir', type=str,
                   help='dir in which simulated polar intensities live')
args = parser.parse_args()
data_dir = args.dir

flist = glob.glob(os.path.join(data_dir, "*q0.18_0.90_Nq*.hdf5"))
s = os.path.basename(flist[0])

model_mastername = re.split('_[0-9]*_downsamp',s)[0]
print ('simulating correlations for the following files')
print flist
master_ouput_fname = os.path.join(data_dir,"%s_all_cors.hdf5"%model_mastername)
print ("saving results in %s"%master_ouput_fname)

#######################
use_basic_mask= True
pmask_basic = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
#######################
meanNorm = False

with h5py.File(master_ouput_fname,'a') as f_out:
    
    # load the water run
    for fname in flist:
        model_name = os.path.basename(fname).split('_downsamp')[0]
        if model_name in f_out.keys():
            print("already has simulated data for %s...Skip!"%model_name)
            continue

        f = h5py.File(fname,'r')

        f_out.create_group(model_name)

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
        # masked_shots = np.zeros_like(shots)
        # meanNorm_shots=np.zeros_like(shots)

        for idx, ss in enumerate(shots):
            if use_basic_mask:
                mask = pmask_basic
            else:
                mask = make_mask(ss,zero_sigma=0.0)
            ss *=mask
            # masked_shots[idx] = ss

            mean_ss = ss.sum(-1)/mask.sum(-1) 

            ss = ss-mean_ss[:,None]
            
            # meanNorm_shots [idx] = np.nan_to_num(ss/mean_ss[:,None]*mask)
            norm_shots[idx] = np.nan_to_num(ss*mask)

        #clean up a bit
        del shots

        diff_norm = norm_shots[1::2]-norm_shots[::2]
        # diff_masked = masked_shots[1::2]-masked_shots[::2]
        # diff_meanNorm = meanNorm_shots[1::2]-meanNorm_shots[::2]

        qs = f['q_intervals'].value[1:,0]
        dc = DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
        mask_ac=dc.autocorr().mean(0)

        dc = DiffCorr(diff_norm,qs,0,pre_dif=True)
        ac = dc.autocorr().mean(0)/mask_ac
        f_out.create_dataset('%s/norm_corr'%model_name,data=ac)

        del diff_norm

        # dc = DiffCorr(diff_masked,qs,0,pre_dif=True)
        # ac = dc.autocorr().mean(0)/mask_ac
        # f_out.create_dataset('masked_corr',data=ac)

        # dc = DiffCorr(diff_meanNorm,qs,0,pre_dif=True)
        # ac = dc.autocorr().mean(0)/mask_ac
        # f_out.create_dataset('meanNorm_corr',data=ac)
        if 'qvalues' not in f_out.keys():
            f_out.create_dataset('qvalues', data=qs)
            f_out.create_dataset('wavlen_in_angstrom', data=f['wavlen_in_angstrom'].value)
        f.close()
print ("Done!")


