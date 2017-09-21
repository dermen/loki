import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma

import argparse
import numpy as np

import matplotlib.pyplot as plt

# load the water run
f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q/run94.tbl','r')
f_out = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/run94_dif_int_basic_pmask_fd.h5','w')
#######################
use_basic_mask= True
pmask_basic = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
#######################
PI = f['polar_imgs']

# get pulse energy, max pos, max height
print("getting pulse energy per shot...")
pulse_energy =np.nan_to_num( \
(f['gas_detector']['f_21_ENRC'].value + f['gas_detector']['f_22_ENRC'].value)/2.)

# extract radial profile max and max pos
print("getting rad prof max pos and max height vals...")
num_shots = f['radial_profs'].shape[0]
max_val = np.zeros(num_shots)
max_pos = np.zeros(num_shots)
for idx in range(num_shots):
    y = f['radial_profs'][idx]
    y_interp = smooth(y, beta=0.1,window_size=50)
    max_val[idx]=y_interp.max()
    max_pos[idx]=y_interp.argmax()
# cluster by pulse energy

print("clustering by pulse energy...")
bins = np.histogram(pulse_energy, bins=200)
pulse_energy_clusters = np.digitize(pulse_energy,bins[1])

print "number of clusters: %d" % len ( list(set(pulse_energy_clusters) ) )
unique_clusters = np.array( sorted( list(set(pulse_energy_clusters)) ) )
#do not use shots when pulse energy is too low
pulse_treshold = 1.5 #mJ
for cc in unique_clusters:
    mean_pulse = np.mean ( pulse_energy[pulse_energy_clusters==cc] )
    if mean_pulse<pulse_treshold:
        continue
    else:
        cluster_to_use = unique_clusters[unique_clusters>=cc]
        break

# sub cluster by max height
shot_set_num = 0

print("sorting shots into clusters...")
for cluster_num in cluster_to_use:
    shot_tags = np.where(pulse_energy_clusters==cluster_num)[0]
    if len(shot_tags)<2:
        print "skipping big cluster %d"%cluster_num
        continue

    cluster_max_vals = max_val[pulse_energy_clusters==cluster_num]
    cluster_max_pos = max_pos[pulse_energy_clusters==cluster_num]
    
    
    bins = np.histogram(cluster_max_vals,bins='fd')
    num_shots = np.where(pulse_energy_clusters==cluster_num)[0].shape[0]
    print "number of shots in cluster: %d"% num_shots
    max_val_clusters = np.digitize(cluster_max_vals,bins[1])
    unique_clusters = np.array(sorted(list(set(max_val_clusters))) )

    for cc in unique_clusters:
        cluster_shot_tags = shot_tags[max_val_clusters==cc]
        if len(cluster_shot_tags)<2:
            print "skipping little cluster %d in big cluster %d"%(cc,cluster_num)
            continue
        
        order = np.argsort(cluster_shot_tags)
        shots = PI[sorted(cluster_shot_tags)]
        max_pos_set = cluster_max_pos[max_val_clusters==cc][order]

        # mask and normalize the shots
        if shots.dtype != 'float64':
            # shots need to be float64 or more. 
            # float32 resulted in quite a bit of numerical error 
            shots = shots.astype(np.float64)
        
        norm_shots = np.zeros_like(shots)
        masked_shots = np.zeros_like(shots)
        for idx, ss in enumerate(shots):
            if use_basic_mask:
                mask = pmask_basic
            else:
                mask = make_mask(ss,zero_sigma=0.0)
            ss *=mask
            masked_shots[idx] = ss

            mean_ss = ss.sum(-1)/mask.sum(-1) 

            ss = ss-mean_ss[:,None]
            norm_shots[idx] = np.nan_to_num(ss*mask)

        #clean up a bit
        del shots

        # rank by max pos and pair
        order = np.argsort(max_pos_set)
        sorted_shots = norm_shots[order]
        if sorted_shots.shape[0]%2>0:
            sorted_shots = sorted_shots[:-1]
        diff_norm = sorted_shots[1::2]-sorted_shots[::2]

        sorted_shots = masked_shots[order]
        if sorted_shots.shape[0]%2>0:
            sorted_shots = sorted_shots[:-1]
        diff_masked = sorted_shots[1::2]-sorted_shots[::2]

        # save difference int
        f_out.create_dataset('norm_diff_%d'%shot_set_num, data = diff_norm)
        f_out.create_dataset('masked_diff_%d'%shot_set_num, data = diff_masked)
        shot_set_num+=1

f_out.close()



