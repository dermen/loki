# lcls cxi jun 207 beam time specific helper functions

import numpy as np
import h5py

from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
from loki.utils import stable

def ave_all_cor(files, dataset = 'ave_norm_corr',
               phi_offset = 0):
    """
    Average quantities across different h5 files, weighted by the number of shots per file

    files - list of str,paths to the files containing quantities to average
    dataset - str, name of the data set to average in the h5 files
    phi_offset - int, number of pixels in phi to ignore on the two extremes of the correlations
    Ignoring these pixels means we are not looking at the very high values near 0 and pi

    """
    n_shots = []
    corrs = []
    for ff in files:
        h5 = h5py.File(ff,'r')
        n_shots.append(h5['num_shots'].value)

        num_phi = h5[dataset].shape[-1]

        try:
            corrs.append(h5[dataset].value[:,phi_offset:num_phi/2-phi_offset])
        except KeyError:
            continue
    corrs = np.array(corrs)
    n_shots = np.array(n_shots,dtype=float)
    
    num_shots = n_shots.sum().astype(float)
    n_shots/=num_shots
    
    ave_corr = (corrs * n_shots[:,None,None]).sum(0)
    return ave_corr, num_shots

def make_streak_mask(shots, num_bins=100):
    """makes a mask to cover streaks in the shots, using statitistics gather on outliers in each shots
    locate outliers in each shot and find two regions where outliers tend to occur and mask these regions

    shots - numpy.array, Nshot*Nq*Nphi, usually a sample for a experimental run.
    num_bins - int, num of bins used for outlier pixel positions histogram
    """    
    # find all the outlier positions
    outlier_pos=[]
    for ss in shots:
        outlier_pos.extend(np.where( is_outlier(ss) )[0])
    outlier_pos = np.array(outlier_pos)
    # statitistics
    bins=np.histogram( outlier_pos, bins=num_bins)
    edges=(bins[1][1:]+bins[1][:-1])/2
    floor = int(shots.shape[0]/num_bins)
    
    # find the peaks
    cutoffs=bins[0]>floor
    indices = np.nonzero(cutoffs[1:] != cutoffs[:-1])[0] + 1
    b = np.split(edges, indices)
    b = np.array( b[0::2] if cutoffs[0] else b[1::2] )
    
    # chunks of indices to mask
    chunks = [range(int(np.floor(bb[0])),int(np.ceil(bb[-1]) )+1 ) for bb in b]
    
    streak_mask=np.ones_like(shots[0])
    for cc in chunks:
        streak_mask[cc] =  0
    return streak_mask.astype(bool)

def corr_pair_diff_PI(norm_shots, mask_corr, qs, 
    qidx_pair = 25,
    phi_offset=10):
    """
    Pair polar intensities at one q value using the corr pair method
    Correlations of individual intensities are computed. Shots with similar correlations are paired
    Pairing metrics are euclidean distances between single-shot correlations
    Pairing done by the Stable roommate method

    norm_shots - numpy.array, Nshot*Nq*Nphi, normalized and zeroed polar intensities
    mask_corr - numpy.array, Nq*Nphi, correlations of the mask used for the PI shots
    qs - numpy.array, Nq, q values covered by the PI shots 
    qidx_pair - int, idx of the q values at which to pair shots
    phi_offset - int, number of pixels in phi to ignore on the two extremes of the correlations
    Ignoring these pixels means we are not looking at the very high values near 0 and pi

    """

    print("doing corr pairing...")
    
    num_phi=norm_shots.shape[-1]
    
    dc = DiffCorr(norm_shots,
      qs,0,pre_dif=True)
    corr = dc.autocorr()
    
    corr/=mask_corr
    corr=corr[:,:,phi_offset:num_phi/2-phi_offset]
    
    
    eps = distance.cdist(corr[:,qidx_pair],corr[:,qidx_pair], metric='euclidean')
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
    diff_norm = np.zeros( (norm_shots.shape[0]/2, 
        1, 
        norm_shots.shape[-1]), 
        dtype=np.float64 )

    for index, pp in enumerate( pairing ):
        diff_norm[index,0] = norm_shots[pp[0],qidx_pair]-norm_shots[pp[1],qidx_pair]

    return diff_norm