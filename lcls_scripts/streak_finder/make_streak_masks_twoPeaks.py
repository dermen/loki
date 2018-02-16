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

def make_streak_mask2(test_shot,mask, num_bins=100):
    masked_average = np.sum(test_shot*mask)/mask.sum()
    masked_std = np.sqrt(np.sum( (test_shot- masked_average)**2*mask)/mask.sum() )

    outliers=np.where(test_shot>masked_average+masked_std*1)[0]

    res=int(test_shot.size/num_bins)/2
    num,bins =np.histogram(outliers,bins=num_bins)
    labels=np.digitize(outliers,bins)
    unique_labels=list(set(labels))
    # print outliers[labels==10]

    masked_ranges=[]

    for ll in unique_labels:
        points = outliers[labels==ll]
        if len(points)==0:
            continue
        else:
            ss = np.max( (int(np.min(points))-res ,0 ) )
            ee = np.min( (int(np.max(points) )+res, test_shot.size) )
            masked_ranges.append( range(ss,ee) )

    streak_mask=np.ones_like(test_shot)
    for cc in masked_ranges:
        cc=np.array(cc)
        cc=cc[cc<streak_mask.size]
        streak_mask[cc] =  0
    return streak_mask.astype(bool)


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


parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/',
                   help='where to look for the polar data')

parser.add_argument('-s','--streak_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/streaks/twoPeaks/',
                   help='where to look for the two peak streaks data')

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

run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')
PI = f['polar_imgs']


streak_dir=args.streak_dir
streak_file = run_file.replace('.tbl','_streaks.h5')
print os.path.join(streak_dir, os.path.join(sample,streak_file))
f_streak=h5py.File(os.path.join(streak_dir, os.path.join(sample,streak_file) ),'a')


shot_tags=f_streak['shot_tags'].value
peak_center=f_streak['all_streak_centers'].value
streak_center=f_streak['all_streak_centers'].value

dists=np.abs(streak_center[:,0]-streak_center[:,1])
two_peak_tags=shot_tags[np.logical_and(dists>174,dists<184)]
two_peak_shots=PI[two_peak_tags,0,:]
two_peak_centers=streak_center[np.logical_and(dists>174,dists<184)]

# cluster by peal positions
res=3
x= two_peak_centers.min(-1)
num_bins = int((x.max()-x.min())/res)
nums, bins=np.histogram(x,bins=50)
labels=np.digitize(x, bins=bins)
unique_labels = np.array(list(set(labels)) )

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')
mask=mask[0,:]
mask=mask.astype(bool)




print('making streak masks...')
all_streak_masks = np.zeros( (unique_labels.size,mask.shape[-1]), dtype=bool )
new_labels = np.zeros_like(labels)
for idx, ll in enumerate(unique_labels):

    test_shot = two_peak_shots[labels==ll].mean(0)
    new_labels[labels==ll] = idx
    all_streak_masks[idx] = streak_mask=make_streak_mask2(test_shot,mask)
if 'two_streaks_tags' in f_streak.keys():
    f_streak.__delitem__('two_streaks_tags')
f_streak.create_dataset('two_streaks_tags', data = two_peak_tags)
try:
    f_streak.create_dataset('two_streak_mask_labels',data=new_labels)
    
except RuntimeError:
    print("soem data sets from previous runs were preserve..")

try:
    f_streak.create_dataset('all_two_streak_masks',data=all_streak_masks)
    
except RuntimeError:
    print("soem data sets from previous runs were preserve..")


f_streak.close()
