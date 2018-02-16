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

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/',
                   help='where to look for the polar data')

parser.add_argument('-sa','--save_autocorr', type=bool, default = False,
                    help='if True save all the individual auto corrs')

parser.add_argument('-s','--streak_dir', type=str, required=True,
                    help='where to look for the two peak streaks data')

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

def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)

def consolidate_datasets(start_str, f_out):
    all_data=[]
    keys = [kk for kk in f_out.keys() if kk.startswith(start_str)]
    if len(keys)==0:
        print("this data set that starts with %s does not exist."%start_str)
        return

    for key in keys:
        all_data.append(f_out[key].value)

        f_out.__delitem__(key)

    all_data=np.concatenate(all_data)
    return all_data


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


# now combine basic mask and streak mask
########
#dir to find the streak masks
streak_dir = os.path.join( args.streak_dir, sample)
########

try:
    streak_file = run_file.replace('.tbl','_streaks.h5')

    # print os.path.join(streak_dir,streak_file)
    f_streak=h5py.File(os.path.join(streak_dir,streak_file),'r')

    all_streak_masks=f_streak['all_two_streak_masks'].value
    all_shot_tags=np.array(sorted(f_streak['two_streaks_tags'].value))
    streak_mask_labels = f_streak['two_streak_mask_labels'].value
    unique_labels=np.unique(streak_mask_labels)

    print("loaded streak mask from %s"%os.path.join(streak_dir,streak_file))

except:
    print("streak masks have not been made for run %d"%run_num)
    sys.exit()

qs = np.linspace(0,0.88,35)

# random pairing no clustering
a = all_shot_tags.shape[0]/2
random_pairs = np.array([[all_shot_tags[:a][ii]
          ,all_shot_tags[a:a*2][ii] ]  for ii in range(a)])

print("random pairing, disregarding streak clusters, to create bench mark dataset")
chunk_size = 5000
num_chunks = int(random_pairs.shape[0]/chunk_size)+1
norm_corrs=[]
shot_nums_per_set=[]
shot_set_num=0

for nc in range(num_chunks):
    print("pairing and computing diff cor for chunk %d..."%nc)
    chunk_pairs = random_pairs[nc*chunk_size:(nc+1)*chunk_size,:]

    random_diff_norm = np.zeros( (chunk_pairs.shape[0],PI.shape[1], PI.shape[-1]), 
        dtype=np.float64) 
    random_diff_streak_masks = np.zeros( (random_diff_norm.shape[0] ,
    random_diff_norm.shape[-1] ),dtype=bool)

    for chunk_idx, pp in enumerate(chunk_pairs):
        shot1=PI[pp[0]]
        shot2=PI[pp[1]]
        # mask and normalize the shots

        mask1 =mask.copy()
        sm_mask1 = all_streak_masks[streak_mask_labels[np.where(all_shot_tags==pp[0])[0][0]]]
        mask1[:10,:]=mask[:10,:]*sm_mask1[None,:]


        mask2 =mask.copy()
        sm_mask2 = all_streak_masks[streak_mask_labels[np.where(all_shot_tags==pp[1])[0][0]]]
        mask2[:10,:]=mask[:10,:]*sm_mask2[None,:]

        sm_mask12=sm_mask1*sm_mask2

        norm_shot1=normalize_shot(shot1, mask1)
        norm_shot2=normalize_shot(shot2, mask2)

        diff_shot12=norm_shot1 - norm_shot2
        random_diff_norm[chunk_idx] =  diff_shot12
        random_diff_streak_masks[chunk_idx] = sm_mask12

    dc = DiffCorr(random_diff_norm, qs, 0,pre_dif=True)
    
    all_diff_masks=np.array([mask]*random_diff_norm.shape[0])
    all_diff_masks[:,:10,:]= random_diff_streak_masks[:,None,:]*all_diff_masks[:,:10,:]
    
    mask_dc= DiffCorr(all_diff_masks.copy(),qs,0,pre_dif=True)
    mask_corr = mask_dc.autocorr()
    print "mask corr shape:"
    print mask_corr.shape

    ac = dc.autocorr() / mask_corr
    norm_corrs.append(ac.mean(0))

    if args.save_autocorr:
        f_out.create_dataset('autocorr_random_%d'%shot_set_num, data = ac)
    
    shot_set_num+=1
    shot_nums_per_set.append(random_diff_norm.shape[0])

    del random_diff_norm

norm_corrs=np.array(norm_corrs)
ave_norm_corr = (norm_corrs * \
    (np.array(shot_nums_per_set)/float(np.sum(shot_nums_per_set)))[:,None,None]).sum(0)


f_out.create_dataset('random_pair_ave_norm_corr',data=ave_norm_corr)
f_out.create_dataset('random_pair_num_shots',data=np.sum(shot_nums_per_set))
f_out.create_dataset('qvalues',data=qs)
f_out.create_dataset('random_pairs',data=random_pairs)

if args.save_autocorr:
    all_random_autocorr = consolidate_datasets('autocorr_random_', f_out)
    f_out.create_dataset('all_random_autocorr', data=all_random_autocorr)



##################################
# We no divide by clusters of two peak shots 
# and pair within to see if there is any 
# difference from random pair 
#
##################################

norm_corrs = []
shot_nums_per_set = []
if args.save_autocorr:
    f_out.create_group('cluster_pair_autocorrs')
f_out.create_group('cluster_pairing')


for ll in unique_labels:
    cluster_shot_tags = all_shot_tags[streak_mask_labels==ll]
    this_streak_mask = all_streak_masks[ll]
    this_mask = mask.copy()
    this_mask[:10,:] = mask[:10,:] * this_streak_mask[None,:]

    if cluster_shot_tags.size<2:
        print("skipping cluster %d"%ll)
        continue
    if cluster_shot_tags.size%2 == 1 :
        cluster_shot_tags=cluster_shot_tags[:-1]

    shots = PI[sorted(cluster_shot_tags)]

    print "number of shots in pairing cluster: %d"% len(cluster_shot_tags)
        # mask and normalize the shots
    if shots.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        shots = shots.astype(np.float64)
    
    norm_shots = np.zeros_like(shots)
    
    for idx, ss in enumerate(shots):
        
        norm_shots[idx] = normalize_shot(ss, this_mask)

    #clean up a bit
    del shots

    #random pairing within cluster
    a = cluster_shot_tags.shape[0]/2
    diff_pair = np.array([[cluster_shot_tags[:a][ii]
          ,cluster_shot_tags[a:a*2][ii] ]  for ii in range(a)])
    diff_norm=norm_shots[:a]-norm_shots[a:a*2]

    #diff corr
    dc = DiffCorr(diff_norm, qs, 0,pre_dif=True)

    mask_dc= DiffCorr(this_mask[None,:,:],qs,0,pre_dif=True)
    mask_corr = mask_dc.autocorr()
    print "mask corr shape:"
    print mask_corr.shape

    ac = dc.autocorr() / mask_corr
    norm_corrs.append(ac.mean(0))

    f_out.create_dataset('cluster_pairing/pairing_cluster_%d'%ll, data = diff_pair)
    if args.save_autocorr:
        f_out.create_dataset('cluster_pair_autocorrs/autocorr_cluster_%d'%ll, data = ac)

    shot_nums_per_set.append(diff_norm.shape[0])

    del norm_shots
print("***************averaging all diff corrs...")
norm_corrs=np.array(norm_corrs)
ave_norm_corr = (norm_corrs * \
    (np.array(shot_nums_per_set)/float(np.sum(shot_nums_per_set)))[:,None,None]).sum(0)

f_out.create_dataset('cluster_pair_num_shots',data=np.sum(shot_nums_per_set))
f_out.create_dataset('cluster_pair_ave_norm_corr',data=ave_norm_corr)

f_out.close()
