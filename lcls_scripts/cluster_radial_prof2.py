import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
import os

import numpy.ma as ma

import argparse
import numpy as np

import matplotlib.pyplot as plt


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
out_file = run_file.replace('.tbl','_cor.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')


if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')

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
shot_nums_per_set = []
norm_corrs = []
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

        for idx, ss in enumerate(shots):

            ss *=mask

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


        # dummy qvalues
        qs = np.linspace(0.1,1.0, diff_norm.shape[1])
        dc = DiffCorr(diff_norm, qs, 0,pre_dif=True)
        ac = dc.autocorr()
        norm_corrs.append(ac.mean(0))
        shot_nums_per_set.append(diff_norm.shape[0])
    
        # save difference int
        f_out.create_dataset('autocorr_%d'%shot_set_num, data = ac)

        shot_set_num+=1


ave_norm_corr = (norm_corrs * \
    (np.array(shot_nums_per_set)/float(np.sum(shot_nums_per_set)))[:,None,None]).sum(0)

qs = np.linspace(0.1,1.0, diff_norm.shape[1])
mask_dc = DiffCorr(mask[None,:], qs, 0, pre_dif=True)
mask_corr = mask_dc.autocorr().mean(0)
ave_norm_corr /= mask_corr

f_out.create_dataset('ave_norm_corr',data=ave_norm_corr)
f_out.create_dataset('num_shots',data=np.sum(shot_nums_per_set))

f_out.close()

# combine all the diffcorr into one file

f = h5py.File(os.path.join(save_dir, out_file),'r')

comb_file = out_file.replace('_cor.h5','_all_diffcorr.h5') 

f_out = h5py.File(os.path.join(save_dir, comb_file),'w')

keys = [kk for kk in f.keys() if kk.startswith('autocorr')]

all_diff_cor = []
for kk in keys:
    all_diff_cor.append(f[kk].value)

all_diff_cor = np.concatenate(all_diff_cor)

f_out.create_dataset('diff_corr', data = all_diff_cor)
f_out.close()
f.close()


