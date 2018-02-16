import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
from loki.utils import stable
import os
import sys
from loki.make_tag_pairs import MakeTagPairs

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

parser.add_argument('-sa','--save_autocorr', type=bool, default = False,
                    help='if True save all the individual auto corrs')

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


def pair_diff_PI(norm_shots, interp_rps, qs):
    print("doing corr pairing...")
    #dummy qs
    num_phi=norm_shots.shape[-1]
    
    eps = distance.cdist(interp_rps,interp_rps, metric='euclidean')
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
        qs.size, 
        norm_shots.shape[-1]), 
        dtype=np.float64 )

    for index, pp in enumerate( pairing ):
        diff_norm[index] = norm_shots[pp[0]]-norm_shots[pp[1]]

    return diff_norm, pairing


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
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)[:4,:]
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')[:4,:]

PI = f['polar_imgs']


# now combine basic mask and streak mask
########
#dir to find the streak masks
streak_dir = os.path.join( '/reg/data/ana14/cxi/cxilp6715/scratch/streaks/kmeans/', sample)
########

try:
    streak_file = run_file.replace('.tbl','_streaks.h5')
    # print os.path.join(streak_dir,streak_file)
    f_streak=h5py.File(os.path.join(streak_dir,streak_file),'r')

    all_streak_masks=f_streak['all_streak_masks'].value
    all_shot_tags=np.array(sorted(f_streak['shot_tags'].value))
    streak_mask_labels = f_streak['streak_masks_labels'].value

    print("loaded streak mask from %s"%os.path.join(streak_dir,streak_file))

except:
    print("streak masks have not been made for run %d"%run_num)
    sys.exit()

qs = np.array([0.2,0.22,0.24,0.26])


# extract radial profile max and max pos
print("getting rad prof max height vals, pos, and interpolating rad profs...")
num_shots = all_shot_tags.shape[0]

interp_rps = np.zeros( (num_shots,f['radial_profs'].shape[-1]) )
max_pos = np.zeros(num_shots)
for idx in range(num_shots):
    y = f['radial_profs'][all_shot_tags[idx]]
    y_interp = smooth(y, beta=0.1,window_size=30)

    interp_rps[idx]=y_interp
    max_pos[idx] = y_interp.argmax()

# cluster by pulse energy

print("clustering by average shot energy..")
pulse_energy = PI[sorted(all_shot_tags),0,:].mean(-1)
bins = np.histogram(pulse_energy, bins=200)
pulse_energy_clusters = np.digitize(pulse_energy,bins[1])

print "number of clusters: %d" % len ( list(set(pulse_energy_clusters) ) )
cluster_to_use = np.array( sorted( list(set(pulse_energy_clusters)) ) )

# sub cluster by max height
shot_set_num = 0
norm_corrs = []
shot_nums_per_set = []

print("sorting shots into clusters...")
for cluster_num in cluster_to_use:
    shot_tags = all_shot_tags[pulse_energy_clusters==cluster_num]
    if len(shot_tags)<2:
        print "skipping big cluster %d"%cluster_num
        continue

    
    num_shots = shot_tags.shape[0]
    print "number of shots in cluster: %d"% num_shots
    
    cluster_max_pos = max_pos[pulse_energy_clusters==cluster_num]
    rad_profs=interp_rps[pulse_energy_clusters==cluster_num]

    sm_labels = streak_mask_labels[pulse_energy_clusters==cluster_num]

    bins = np.histogram(cluster_max_pos,bins='fd')
    max_pos_clusters = np.digitize(cluster_max_pos,bins[1])
    unique_clusters = np.array(sorted(list(set(max_pos_clusters))) )

    for cc in unique_clusters:
        cluster_shot_tags = shot_tags[max_pos_clusters==cc]
        cluster_sm_labels = sm_labels[max_pos_clusters==cc]


        if len(cluster_shot_tags)<2:
            print "skipping little cluster %d in big cluster %d"%(cc,cluster_num)
            continue

        order = np.argsort(cluster_shot_tags)
        shots = PI[sorted(cluster_shot_tags)][:,:4,:]

        rad_profs_set = rad_profs[max_pos_clusters==cc][order]
        print rad_profs_set.shape, shots.shape

        print "number of shots in pairing cluster: %d"% len(cluster_shot_tags)
        # mask and normalize the shots
        if shots.dtype != 'float64':
            # shots need to be float64 or more. 
            # float32 resulted in quite a bit of numerical error 
            shots = shots.astype(np.float64)
        
        norm_shots = np.zeros_like(shots)
        
        for idx, ss in enumerate(shots):
            this_streak_mask = all_streak_masks[cluster_sm_labels[idx]]
            this_mask = mask*this_streak_mask[None,:]
            ss *=this_mask
            
            mean_ss = ss.sum(-1)/this_mask.sum(-1) 

            ss = ss-mean_ss[:,None]
            norm_shots[idx] = np.nan_to_num(ss*this_mask)

        #clean up a bit
        del shots
        # for sanity check only
        if norm_shots.shape[0]%2>0:
            norm_shots = norm_shots[:-1]
            rad_profs_set=rad_profs_set[:-1]
            cluster_shot_tags=sorted(cluster_shot_tags)[:-1]

            cluster_sm_labels=cluster_sm_labels[:-1]
        # diff_norm=norm_shots[::2]-norm_shots[1::2]


        diff_norm, pairing = pair_diff_PI(norm_shots, rad_profs_set,qs)
        diff_pair = np.zeros( (diff_norm.shape[0] , 2 ))
        diff_streak_masks = np.zeros( (diff_norm.shape[0] , diff_norm.shape[-1] ),dtype=bool)

        for index, pp in enumerate( pairing ):
            diff_pair[index,0] = cluster_shot_tags[pp[0]]
            diff_pair[index,1] = cluster_shot_tags[pp[1]]

            sm1 = all_streak_masks[cluster_sm_labels[pp[0]]]
            sm2 = all_streak_masks[cluster_sm_labels[pp[1]]]
            diff_streak_masks[index] = sm1*sm2


        dc = DiffCorr(diff_norm, qs, 0,pre_dif=True)
        all_diff_masks=np.array([mask]*diff_norm.shape[0])
        all_diff_masks[:,:4,:]= diff_streak_masks[:,None,:]*all_diff_masks[:,:4,:]
        mask_dc= DiffCorr(all_diff_masks.copy(),qs,0,pre_dif=True)
        mask_corr = mask_dc.autocorr()
        print "mask corr shape:"
        print mask_corr.shape

        ac = dc.autocorr() / mask_corr
        norm_corrs.append(ac.mean(0))

        f_out.create_dataset('pairing_%d'%shot_set_num, data = diff_pair)
        if args.save_autocorr:
            f_out.create_dataset('autocorr_%d'%shot_set_num, data = ac)

        shot_set_num+=1
        shot_nums_per_set.append(diff_norm.shape[0])
        # save difference int
        # f_out.create_dataset('norm_diff_%d'%shot_set_num, data = diff_norm)
        # 
        # shot_set_num+=1
    ##############
    # Dubgging
    # break
    ##############
ave_norm_corr = (norm_corrs * \
    (np.array(shot_nums_per_set)/float(np.sum(shot_nums_per_set)))[:,None,None]).sum(0)


f_out.create_dataset('ave_norm_corr',data=ave_norm_corr)
f_out.create_dataset('num_shots',data=np.sum(shot_nums_per_set))
f_out.create_dataset('qvalues',data=qs)

pair_keys = [kk for kk in f_out.keys() if kk.startswith('pairing')]
all_pairing=[]
# Consolidate and delete individual datasets
for key in pair_keys:
  all_pairing.append(f_out[key].value)

  f_out.__delitem__(key)

all_pairing=np.concatenate(all_pairing)
f_out.create_dataset('all_pairings',data=all_pairing)
# print all_pairing

if args.save_autocorr:
    print("cleaning up and closing file...")
    corr_keys = [kk for kk in f_out.keys() if kk.startswith('autocorr')]
    all_corrs=[]
    # Consolidate and delete individual datasets
    for key in corr_keys:
      all_corrs.append(f_out[key].value)

      f_out.__delitem__(key)

    all_corrs=np.concatenate(all_corrs)
    f_out.create_dataset('all_autocorrs',data=all_corrs)

f_out.close()