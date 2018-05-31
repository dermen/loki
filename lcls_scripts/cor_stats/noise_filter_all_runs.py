import h5py
import numpy as np

from sklearn.decomposition import NMF

from scipy.interpolate import interp1d

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

def reorder_data(data, exp_cpsi):
    n_shots=data.shape[0]
    order=np.argsort(exp_cpsi)
    exp_cpsi = np.array(sorted(exp_cpsi))
    for ii in range(n_shots):
        data[ii] = data[ii][order]
    return data, exp_cpsi

def interp_shots(norm_X2, interp_num_phi,old_cpsi,new_cpsi):
    
    interp_X = np.zeros( (norm_X2.shape[0],interp_num_phi) )

    for ii in range(interp_X.shape[0]):
        f = interp1d(old_cpsi, norm_X2[ii])
        try:
            interp_X[ii] = f(new_cpsi)
        except:
            print ii
            break
    return interp_X



# get 18k shots from each run and store the average, norm_average, and respective errors, save them
output_fname='/reg/d/psdm/cxi/cxilp6715/results/noise_reduction_nnmf/GDP_diff.h5'
phi_offset = 30
num_phi = 354
interp_num_phi =100
# load simulations used for fitting
sims = np.load('/reg/d/psdm/cxi/cxilp6715/results/noise_reduction_nnmf/1gp2_closed_48sim.npy')

all_exp_cpsi = np.load('/reg/d/psdm/cxi/cxilp6715/results/noise_reduction_nnmf/exp_cpsi.npy')
all_sim_cpsi = np.load('/reg/d/psdm/cxi/cxilp6715/results/noise_reduction_nnmf/sim_cpsi.npy')

run_nums=[18,30,31,32,37,38,40,41]
buf_run_nums=[10,11,13,15]
with h5py.File(output_fname,'w') as f_out:
    f_out.create_dataset('phi_offset',data=phi_offset)
    f_out.create_dataset('sims', data = sims)

    for idx, buf_r_num in enumerate(buf_run_nums):
        pro_r_num=run_nums[idx*2:(idx+1)*2]
        print buf_r_num, pro_r_num

        grp = f_out.create_group('Pro_run%d_%d_Buf_run%d'%(pro_r_num[0],pro_r_num[1],buf_r_num))
        # noised filtered average shot, and errors
        pro=np.zeros( (35, interp_num_phi), dtype=np.float64)
        err = np.zeros( (35, interp_num_phi), dtype=np.float64)
        interp_cpsi = np.zeros( (35, interp_num_phi), dtype=np.float64)

        for qidx in range(35):
            # data and simulations
            GDPpro = get_shots(pro_r_num[0],'GDP_pro',qidx, num_shots=18000)[:,phi_offset:num_phi/2-phi_offset]
            GDPpro1 = get_shots(pro_r_num[1],'GDP_pro',qidx, num_shots=18000)[:,phi_offset:num_phi/2-phi_offset]
            GDPpro = np.concatenate([GDPpro,GDPpro1])
            print GDPpro.shape

            buf = get_shots(buf_r_num,'GDP_buf',qidx, num_shots=36000)[:,phi_offset:num_phi/2-phi_offset]
            
            X = sims[:,qidx,phi_offset:num_phi/2-phi_offset]

            exp_cpsi = all_exp_cpsi[qidx,phi_offset:num_phi/2-phi_offset]
            sim_cpsi = all_sim_cpsi[qidx,phi_offset:num_phi/2-phi_offset]
            # normalize
            norm_X = normalize_set(X)
            norm_X = 0.5*(norm_X[:,::-1]+norm_X)

            norm_GDPpro = normalize_set(GDPpro)
            norm_buf = normalize_set(buf)

            # reorder
            norm_GDPpro2, exp_cpsi2 = reorder_data(norm_GDPpro.copy(),exp_cpsi.copy())
            norm_buf2, _ = reorder_data(norm_buf.copy(),exp_cpsi.copy())
            norm_X2, sim_cpsi2 = reorder_data(norm_X.copy(),sim_cpsi.copy())
            # interpolate
            
            new_cpsi = np.linspace(np.max( (exp_cpsi2.min(),sim_cpsi2.min()) )+0.05,
                                  np.min((exp_cpsi2.max(), sim_cpsi2.max()))-0.05,
                                  interp_num_phi,endpoint=False )
            interp_cpsi[qidx] = new_cpsi

            interp_X = interp_shots(norm_X2, interp_num_phi, sim_cpsi2, new_cpsi)
            interp_pro = interp_shots(norm_GDPpro2, interp_num_phi, exp_cpsi2, new_cpsi)
            interp_buf = interp_shots(norm_buf2, interp_num_phi, exp_cpsi2, new_cpsi)

            # transform and inverse transform
            model = NMF(n_components=10,solver='cd')
            W=model.fit_transform(interp_X)
            H=model.components_

            new_buf = model.transform(interp_buf)
            new_pro = model.transform(interp_pro)
    
            inverse_diff = model.inverse_transform(new_pro-new_buf)

            # average and error estimate

            pro[qidx] = inverse_diff.mean(0)
            err[qidx] = inverse_diff.std(0)/np.sqrt(inverse_diff.shape[0])

        grp.create_dataset('ave_cor',data=pro)
        grp.create_dataset('err',data=err)
        grp.create_dataset('num_shots',data=inverse_diff.shape[0])
        grp.create_dataset('interp_cpsi', data = interp_cpsi)
        grp.create_dataset('nnmf_n_components', data = model.n_components)

