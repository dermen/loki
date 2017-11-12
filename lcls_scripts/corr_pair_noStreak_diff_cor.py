import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
from loki.utils import stable
from loki.make_tag_pairs import MakeTagPairs
import os

import numpy.ma as ma

import argparse
import numpy as np

def pair_diff_PI(norm_shots, mask_corr, qs, 
    qidx_pair = 25,
    phi_offset=10):
    print("doing corr pairing...")
    #dummy qs
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


def make_streak_mask(shots, num_bins=100):
    
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


parser = argparse.ArgumentParser(description='Compute difference correlation by pairing single intensity correlations.')
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

parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q used for pairing or the only q used for pairing')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q used for pairing or None')

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


# in this script, I will cluster by correlations from single intensities
# the logic is that correlation curves from single intensities will exibihit mostly artifacts

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



# figure which qs are used for pairing
qmin = args.qmin
qmax = args.qmax

if qmax is None:
    qpair_inds = [qmin]
else:
    qpair_inds = range(qmin,qmax+1) # qmax is included

# load q values
qs = np.load(os.path.join( args.data_dir, 'qvalues.npy') )[qpair_inds]

# now combine basic mask and streak mask
print("making streak_mask")
sample_shots = PI[::80,0,:]
num_bins = int(sample_shots.shape[0]/10)
streak_mask=make_streak_mask(sample_shots, num_bins=num_bins)

mask = mask[qpair_inds,:]
mask = mask*streak_mask[None,:]

mask_dc = DiffCorr(mask[None,:], qs, 0, pre_dif=True)
mask_corr = mask_dc.autocorr()

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
norm_corrs = []
shot_nums_per_set = []

print("sorting shots into clusters...")
for cluster_num in cluster_to_use:
    shot_tags = np.where(pulse_energy_clusters==cluster_num)[0]
    if len(shot_tags)<2:
        print "skipping big cluster %d"%cluster_num
        continue

    
    
    num_shots = np.where(pulse_energy_clusters==cluster_num)[0].shape[0]
    print "number of shots in cluster: %d"% num_shots
    
    cluster_max_pos = max_pos[pulse_energy_clusters==cluster_num]
    bins = np.histogram(cluster_max_pos,bins='fd')
    max_pos_clusters = np.digitize(cluster_max_pos,bins[1])
    unique_clusters = np.array(sorted(list(set(max_pos_clusters))) )

    for cc in unique_clusters:
        cluster_shot_tags = shot_tags[max_pos_clusters==cc]
        if len(cluster_shot_tags)<2:
            print "skipping little cluster %d in big cluster %d"%(cc,cluster_num)
            continue

        order = np.argsort(cluster_shot_tags)
        shots = PI[sorted(cluster_shot_tags)][:,qpair_inds,:]
        print "number of shots in pairing cluster: %d"% len(cluster_shot_tags)
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
        # for sanity check only
        if norm_shots.shape[0]%2>0:
            norm_shots = norm_shots[:-1]
        # diff_norm=norm_shots[::2]-norm_shots[1::2]

        diff_norm = np.zeros( (norm_shots.shape[0]/2, 
        len(qpair_inds), 
        norm_shots.shape[-1]), 
        dtype=np.float64 )

        for ii,qidx4pairing in enumerate(qpair_inds):
              diff_norm[:,ii,:] = pair_diff_PI(norm_shots.copy(), mask_corr.copy()
                , qs,
                qidx_pair = qidx4pairing)[:,0,:]
        # dummy qvalues
        dc = DiffCorr(diff_norm, qs[qpair_inds], 0,pre_dif=True)
        ac = dc.autocorr() / mask_corr[:,qpair_inds,:]
        norm_corrs.append(ac.mean(0))
        shot_nums_per_set.append(diff_norm.shape[0])

        f_out.create_dataset('autocorr_%d'%shot_set_num, data = ac)

        shot_set_num+=1
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
f_out.create_dataset('qvalues',data=qs[qpair_inds])

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