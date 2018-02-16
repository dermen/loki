import numpy as np
import h5py
import os

from loki.utils.postproc_helper import *

from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt

import sys
import argparse


def sample_type(x):
    return {-1:'AgB_sml',
    -2:'AgB_wid',
     0:'GDP_buf',
     1:'ALF_buf',
     2:'GDP_pro',
     3:'ALF_pro',
     4:'h2o',
     5:'he',
     6:'3to1_rec_GDP_pro'}[x]

def smooth_unmasked_area(shots, mask):
    smoothed_shots=np.zeros( (shots.shape[0],
        np.where(mask)[0].shape[-1]) )
    for shot_idx in range(shots.shape[0]):
        try:
            smoothed_shots[shot_idx]=smooth(shots[shot_idx][mask],window_size=11)
        except ValueError:
            print shot_idx
            sys.exit()
    return smoothed_shots

def find_highest_two_peaks( smoothed_shots):
    peak_positions_in_pixels=np.zeros( (smoothed_shots.shape[0],2) )
    for shot_idx in range(smoothed_shots.shape[0]):
        peak_pos=find_peaks_cwt(smoothed_shots[shot_idx], range(1,11))
        peak_values=smoothed_shots[shot_idx,peak_pos]
        max_2 = np.argsort(peak_values)[-2:]
        
        peak_positions_in_pixels[shot_idx]=np.where(mask)[0][peak_pos[max_2] ]

    return peak_positions_in_pixels.astype(int)



parser = argparse.ArgumentParser(description='Compute difference correlation by consecutive pairing.')
parser.add_argument('-r','--run', type=int,
                   help='run number')
parser.add_argument('-t','--samp_type', type=int,
                   help='type of data/n \
# Sample IDs\n\
# -1: Silver Behenate smaller angle\n\
# -2: Silver Behenate wider angle\n\
# 0: GDP buffer\n\
# 1: ALF BUffer\n\
# 2: GDP protein\n\
# 3: ALF protein\n\
# 4: Water \n\
# 5: Helium\n\
# 6: 3-to-1 Recovered GDP')

parser.add_argument('-o','--out_dir', type=str,default = None,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/',
                   help='where to look for the polar data')


args = parser.parse_args()

# in this script, I will cluster by 1st PC of the radial profile and then pair by 2nd PC
run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)

# import run file


data_dir = args.data_dir
if args.out_dir is None:
    save_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/rp_clusters/dif_cor/%s'%sample
else:
    save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_streaks.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')


if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
mask=mask[0,:]
mask=mask.astype(bool)

PI = f['polar_imgs']

qs = np.array([0.2])

# find streaks
# get pulse energy, max pos, max height
print("getting pulse energy per shot...")
ave_shot_energy =PI[:,:1].mean(-1).mean(-1)

energy_treshold = 25000 # arbitrary units after polar interpolation 
# shots going to be used
tags = np.array(sorted(np.where(ave_shot_energy>energy_treshold)[0]))

# divide and find streaks
chunk_size=5000
num_chunks = int(tags.size/chunk_size)+1

print('finding streaks...')
for nc in range(num_chunks):
    chunk_tags=tags[nc*chunk_size:(nc+1)*chunk_size]
    # print chunk_tags
    shots=PI[sorted(chunk_tags)][:,0,:]
    # make smoothed shot
    smoothed_shots = smooth_unmasked_area(shots, mask)
    # find two peaks
    peak_pos=find_highest_two_peaks(smoothed_shots)
    # save two peak positions


    f_out.create_dataset('streak_centers_%d'%nc, data = peak_pos)

# consolidate chunks
center_keys = [kk for kk in f_out.keys() if kk.startswith('streak_centers')]

all_streak_centers=[]
# Consolidate and delete individual datasets
for ii,key in enumerate(center_keys):
  all_streak_centers.append(f_out[key].value)
  
  f_out.__delitem__(key)

all_streak_centers=np.concatenate(all_streak_centers)


f_out.create_dataset('all_streak_centers',data=all_streak_centers)
f_out.create_dataset('shot_tags',data=tags)

# some stats about the streaks
print('doing stats with the results...')

dists=np.abs(all_streak_centers[:,0]-all_streak_centers[:,1])
dists=np.array(dists)
two_peak_shot_tags=tags[np.logical_and(dists>175,dists<183)]
f_out.create_dataset('streak_dists',data=dists)
f_out.create_dataset('two_streaks_tags',data=two_peak_shot_tags)


f_out.close()
