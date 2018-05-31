import h5py
import numpy as np

from loki.RingData import DiffCorr


def get_shots(run_num, sample,qidx, num_shots=3500):
    f = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/denoise_polar_intensity/diagnostics/all_pca_maskByRun/%s/run%d_PCA-denoise.h5'%(sample,run_num),
                  'r')
    f_mask = h5py.File('/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q/run%d.tbl'%run_num,'r')

    mask = f_mask['polar_mask_binned'].value
    mask = (mask==mask.max())
    mask.shape
    qs = np.linspace(0,1,mask.shape[0])
    dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
    mask_cor = dc.autocorr().mean(0)

    f_mask.close()
    
    cutoff=f['q%d'%qidx]['num_pca_cutoff'].value
    if num_shots == 'all':
        shots=f['q%d'%qidx]['pca%d'%cutoff]['all_train_difcors'][:]/mask_cor[qidx]
    else:
        shots=f['q%d'%qidx]['pca%d'%cutoff]['all_train_difcors'][:num_shots]/mask_cor[qidx]
    
    return shots[:,0,:]

def normalize(d, zero=True):
    x=d.copy()
    if zero:
        x-=x.min()
    return x/(x.max()-x.min())

def normalize_set(GDPdiff):
    norm_GDP_diff = np.zeros_like(GDPdiff)
    for iq in range(GDPdiff.shape[0]):
        norm_GDP_diff[iq] = normalize(GDPdiff[iq])
    return norm_GDP_diff

# get 18k shots from each run and store the average, norm_average, and respective errors, save them
output_fname='/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/GDP_pros_allshots.h5'
phi_offset = 30
num_phi = 354


run_nums=[18,30,31,32,37,38,40,41]
with h5py.File(output_fname,'w') as f_out:
    f_out.create_dataset('phi_offset',data=phi_offset)
    for r_num in run_nums:
        print r_num
        grp = f_out.create_group('run%d'%r_num)
        
        pro=np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        norm_pro=np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        err = np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        norm_err = np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        
        for qidx in range(35):
        
            GDPpro = get_shots(r_num,'GDP_pro',qidx, num_shots='all')[:,phi_offset:num_phi/2-phi_offset]
            norm_GDPpro = normalize_set(GDPpro)
            
            pro[qidx] = GDPpro.mean(0)
            norm_pro[qidx] = norm_GDPpro.mean(0)
            # print norm_GDPpro.std(0),norm_GDPpro.shape
            norm_err[qidx] = norm_GDPpro.std(0)/np.sqrt(GDPpro.shape[0])
            err[qidx] = GDPpro.std(0)/np.sqrt(GDPpro.shape[0])

        grp.create_dataset('ave_cor',data=pro)
        grp.create_dataset('ave_norm_cor',data=norm_pro)
        grp.create_dataset('err',data=err)
        grp.create_dataset('norm_err',data=norm_err)
        grp.create_dataset('num_shots',data=GDPpro.shape[0])

# now do the same with buffer shots
output_fname='/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/GDP_bufs_allshots.h5'
phi_offset = 30
num_phi = 354


run_nums=[10,11,13,15]
with h5py.File(output_fname,'w') as f_out:
    f_out.create_dataset('phi_offset',data=phi_offset)
    for r_num in run_nums:
        print r_num
        grp = f_out.create_group('run%d'%r_num)
        
        pro=np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        norm_pro=np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        err = np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        norm_err = np.zeros( (35, num_phi/2-2*phi_offset), dtype=np.float64)
        
        for qidx in range(35):
        
            GDPpro = get_shots(r_num,'GDP_buf',qidx, num_shots='all')[:,phi_offset:num_phi/2-phi_offset]
            norm_GDPpro = normalize_set(GDPpro)
            
            pro[qidx] = GDPpro.mean(0)
            norm_pro[qidx] = norm_GDPpro.mean(0)
            # print norm_GDPpro.std(0),norm_GDPpro.shape
            norm_err[qidx] = norm_GDPpro.std(0)/np.sqrt(GDPpro.shape[0])
            err[qidx] = GDPpro.std(0)/np.sqrt(GDPpro.shape[0])

        grp.create_dataset('ave_cor',data=pro)
        grp.create_dataset('ave_norm_cor',data=norm_pro)
        grp.create_dataset('err',data=err)
        grp.create_dataset('norm_err',data=norm_err)
        grp.create_dataset('num_shots',data=GDPpro.shape[0])
