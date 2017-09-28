import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
from loki.utils import stable
from loki.make_tag_pairs import MakeTagPairs
import os

import numpy.ma as ma

import argparse
import numpy as np
import sys

import matplotlib.pyplot as plt



def pair_diff_PI(max_pos_cluster_shots, 
    degree = 15, 
    qidx_pair = 25):
    print("doing cheby pairing...")
    if max_pos_cluster_shots.shape[0]%2>0:
        max_pos_cluster_shots = max_pos_cluster_shots[:-1]
    # cheby pair within each 
    # pairing with chebyfit polynomials
    # I am going to use the qidx = 25 for pairing
    print("fitting to polynomials....")
    fits = np.zeros( (max_pos_cluster_shots.shape[0],
        max_pos_cluster_shots.shape[-1])
        ,dtype = np.float64 )
    for ii in range(max_pos_cluster_shots.shape[0]):
        try:
            _,_,yfit = fit_periodic(max_pos_cluster_shots[ii,qidx_pair].copy(), 
                    mask=np.ones(max_pos_cluster_shots.shape[-1],dtype=bool),
                    deg=degree,overlap=0.1)
            fits[ii] = yfit
        except TypeError:
            # print ii
            # print max_pos_cluster_shots[ii,qidx_pair]
            # sys.exit()
            continue

    
    eps = distance.cdist(fits, fits, metric='euclidean')
    # do this so the diagonals are not the minimum, i.e. don't pair shot with itself
    epsI = 1.1 * eps.max(1) * np.identity(eps.shape[0])
    eps += epsI

    shot_preference = np.roll(eps.argsort(1), 1, axis=1)
    pref_dict = {str(E[0]): list(E[1:])
             for E in shot_preference.astype(str)}

    print("stable roommate pair....")
    pairs_dict = stable.stableroomate(prefs=pref_dict)

    pairing = np.array(MakeTagPairs._remove_duplicate_pairs(pairs_dict) )

    print("computing difference intensities...")
    diff_norm = np.zeros( (max_pos_cluster_shots.shape[0]/2, 
        max_pos_cluster_shots.shape[1], 
        max_pos_cluster_shots.shape[-1]), 
        dtype=np.float64 )

    for index, pp in enumerate( pairing ):
        diff_norm[index] = max_pos_cluster_shots[pp[0]]-max_pos_cluster_shots[pp[1]]

    return diff_norm


# load the simulated shots
qidx4pairing = int(sys.argv[1])
f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/test_simulated_data/test_shots_compact_open_35qs.h5','r')
f_out = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/test_simulated_data/test_shots_compact_open_cehby_corrs.h5','w')
#######################
use_basic_mask= True

pmask_basic = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
#######################
PI = f['polar_imgs']

shot_set_num = 0
norm_corrs = []
shot_nums_per_set = []
num_sets = 50

for n_set in range(num_sets):
    print n_set
    cluster_shot_tags = range(n_set*200,(n_set+1)*200)
    shots = PI[cluster_shot_tags]
    
    # mask and normalize the shots
    if shots.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        shots = shots.astype(np.float64)
    
    norm_shots = np.zeros_like(shots)
    
    for idx, ss in enumerate(shots):
        if use_basic_mask:
            mask = pmask_basic
        else:
            mask = make_mask(ss,zero_sigma=0.0)
        ss *=mask
        
        mean_ss = ss.sum(-1)/mask.sum(-1) 

        ss = ss-mean_ss[:,None]
        norm_shots[idx] = np.nan_to_num(ss*mask)

    #clean up a bit
    del shots


    diff_norm = pair_diff_PI(norm_shots,
            qidx_pair = qidx4pairing)

    # dummy qvalues
    qs = np.linspace(0.1,1.0, diff_norm.shape[1])
    dc = DiffCorr(diff_norm, qs, 0,pre_dif=True)
    ac = dc.autocorr().mean(0)
    norm_corrs.append(ac)
    shot_nums_per_set.append(diff_norm.shape[0])

    # save difference int
    f_out.create_dataset('norm_diff_%d'%shot_set_num, data = diff_norm)
    # 
    shot_set_num+=1
##############
# Dubgging
# break
##############
ave_norm_corr = (norm_corrs * \
    (np.array(shot_nums_per_set)/float(np.sum(shot_nums_per_set)))[:,None,None]).sum(0)
# if use_basic_mask:
#     qs = np.linspace(0.1,1.0, diff_norm.shape[1])
#     mask_dc = DiffCorr(mask[None,:], qs, 0, pre_dif=True)
#     mask_corr = mask_dc.autocorr().mean(0)
#     ave_norm_corr /= mask_corr

f_out.create_dataset('ave_norm_corr',data=ave_norm_corr)
f_out.create_dataset('num_shots',data=np.sum(shot_nums_per_set))

f_out.close()



