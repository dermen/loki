import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

from scipy.signal import argrelmax

from loki.utils.postproc_helper import smooth
#### use this script to gather statistics on the hit rate and radial profiles
run_nums=[]
ratios=[]
for rr in range(1,95):
    data_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q/'
    run_file = "run%d.tbl"%rr
    run_filename= os.path.join(data_dir, run_file)

    if not os.path.isfile(run_filename):
        continue
    run_nums.append(rr)
    f = h5py.File(run_filename, 'r')
    print f.keys()
    rps=f['radial_profs'].value

    beta = 50 # smoothing factor
    window_size = 200 # pixel units
    order = 250 
    smooth_rps=np.zeros_like(rps)
    for ii in range(rps.shape[0]):
        smooth_rps[ii] = smooth(rps[ii], beta=beta, window_size=window_size) 

    pk_pos=[]
    pk_vals=[]
    selected_inds=[]

    for ii,rr in enumerate(smooth_rps):
        mx = argrelmax(rr, order=order)[0] 

        # make sure there is only one peak!
        if not len(mx) == 1:
            continue

        # make sure the peak lies in the desired range.. 
        pp=mx[0]
        pv = rr[pp]
        
        
        if not pk_range[0] < pp < pk_range[1] : 
            
            continue

    #       make sure the peak value is max in the original profile, 
    #       because it was selected using line-subtracted profile

        if not rr[pk_range[0]] < pv and not rr[pk_range[1]] < pv :
            continue 
                
        pk_pos.append(pp)
        pk_vals.append(pv) 
        selected_inds.append(ii)
        
    pk_vals =np.array(pk_vals)
    pk_pos = np.array(pk_pos)
    selected_inds=np.array(selected_inds)

    ratio=selected_inds.size/float(rps.shape[0])
    ratios.append(ratio)
    pk_stats=np.vstack([pk_vals,pk_pos, selected_inds])

run_nums=np.array(run_nums)
ratios=np.array(ratios)

stats=np.vstack([run_nums,ratios])