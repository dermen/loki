import h5py
from loki.RingData import DiffCorr
from loki.utils.postproc_helper import *
from loki.utils import stable
from loki.make_tag_pairs import MakeTagPairs

from loki.RingData.DiffCorr import DiffCorr

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

import sys
import os

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


parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q',
                   help='where to look for the polar data')


parser.add_argument('-c','--cluster_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/scratch/pca_clusters/higher_q/',
                   help='where to look for pca cluster assignment')

parser.add_argument('-m','--pair_metric', type=str, default = 'int',
                   help='which metric to use for clustering/n \
                   int: default, for intensity at each q value/n\
                   corr: use single shot corr at each q value for pairing')

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

def pair_diff_PI(norm_shots, mask_corr, qs, 
    qidx_pair = 25,
    phi_offset=0,
    pair_method='int'):

    if pair_method=='corr':
        print("doing corr pairing...")
        #dummy qs
        num_phi=norm_shots.shape[-1]
        
        dc = DiffCorr(norm_shots,
          qs,0,pre_dif=True)
        corr = dc.autocorr()
        
        corr/=mask_corr
        corr=corr[:,:,phi_offset:num_phi/2-phi_offset]
        
        
        eps = distance.cdist(corr[:,qidx_pair],corr[:,qidx_pair], metric='euclidean')
        
    if pair_method=='int':
        print "doing intensity pair..."
        eps = distance.cdist(norm_shots[:,qidx_pair],norm_shots[:,qidx_pair], metric='euclidean')
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
        norm_shots.shape[1], 
        norm_shots.shape[-1]), 
        dtype=np.float64 )

    for index, pp in enumerate( pairing ):
        diff_norm[index] = norm_shots[pp[0]]-norm_shots[pp[1]]

    return diff_norm, pairing


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
save_dir = os.path.join( args.out_dir, sample)
cluster_dir =os.path.join(  args.cluster_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
cluster_file = run_file.replace('.tbl','_PCA-cluster.h5')
f_cluster = h5py.File(os.path.join(cluster_dir, cluster_file),'r')
cluster_set_keys = f_cluster.keys()

out_file = run_file.replace('.tbl','_cor.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'w')

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    mask = np.load('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy')


qs=np.linspace(0.2,0.88,mask.shape[0])
dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_ac=dc.autocorr()


PI = f['polar_imgs']
shot_tags = np.arange(0,PI.shape[0])

for set_key in cluster_set_keys:
    print("computing diff cor for %s..."%set_key)
    qidx = int( set_key.split('q')[1] )
    labels = f_cluster[set_key]['cluster_labels'].value.astype(int)

    f_out.create_group(set_key)

    unique_labels=np.unique(labels)
    cluster_corrs=[]
    cluster_num_shots=[]
    for ll in unique_labels:
        cluster_shot_tags=shot_tags[labels==ll]
        if cluster_shot_tags.size<2:
            print("Skipping cluster %d for q %d"%(ll, qidx))
            continue

        if cluster_shot_tags.size%2>0:
            cluster_shot_tags=cluster_shot_tags[:-1]
        shots=PI[sorted(cluster_shot_tags),:,:]

        norm_shots=np.zeros_like(shots)
        for ii in range(norm_shots.shape[0]):
            norm_shots[ii] = normalize_shot(shots[ii], mask)
        
        num_shots = norm_shots.shape[0]
        print ("number of shots to pair: %d"%num_shots)


        #cluster by intensity at the q for pairing
        num_bins=int(shots.shape[0]/394)
        total_intensities=shots.sum(-1)[:,qidx]
        nums,bins=np.histogram(total_intensities, bins=num_bins)

        int_cluster_labels = np.digitize(total_intensities,bins)
        int_clusters=np.unique(int_cluster_labels)

        diff_norm=[]
        pairings=[]
        int_cluster_shot_tag= np.arange(0,num_shots)
        for cc in int_clusters:
            cluster_shots=norm_shots[int_cluster_labels==cc]
            tags=int_cluster_shot_tag[int_cluster_labels==cc]

            if cluster_shots.shape[0]<2:
                continue
            if cluster_shots.shape[0]%2>0:
                cluster_shots=cluster_shots[:-1]
            dn, pairing = pair_diff_PI(cluster_shots,
                             mask_ac,qs,
                             qidx_pair=qidx, 
                              pair_method='int')
            tag_pairs = np.zeros_like(pairing,dtype=int)
            for jj,pp in enumerate(pairing):
                tag_pairs[jj]=np.array([tags[pp[0]], tags[pp[1]]])
            pairings.append(tag_pairs)
            diff_norm.append(dn)

        diff_norm=np.concatenate(diff_norm)
        pairings=np.concatenate(pairings)
            
        dc = DiffCorr(diff_norm, qs, 0, pre_dif=True)
        ac = dc.autocorr()/mask_ac
        ac = ac.mean(0)

        cluster_corrs.append(ac)
        cluster_num_shots.append(num_shots)
        f_out.create_dataset('%s/cluster_%d_pairing'%(set_key,ll), data=pairings)
    
    cluster_corrs=np.array(cluster_corrs)
    cluster_num_shots=np.array(cluster_num_shots)

    ave_norm_corr = (cluster_corrs * \
    (cluster_num_shots/float(np.sum(cluster_num_shots)))[:,None,None]).sum(0)

    f_out.create_dataset('%s/cluster_num_shots'%set_key, data=cluster_num_shots)
    f_out.create_dataset('%s/cluster_corrs'%set_key, data=cluster_corrs)
    f_out.create_dataset('%s/ave_norm_corr'%set_key, data=ave_norm_corr)

print("done!")
f_out.close()