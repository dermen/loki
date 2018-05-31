import numpy as np
import h5py

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def normalize(d, zero=True):
    x=d.copy()
    if zero:
        x-=x.min()
    return x/(x.max()-x.min())


def reorder_data(data, exp_cpsi):
    n_q=data.shape[0]
    for ii in range(n_q):
        order=np.argsort(exp_cpsi[ii])
        exp_cpsi[ii] = sorted(exp_cpsi[ii])
        data[ii] = data[ii][order]
    return data, exp_cpsi

def _weighted_sum(X, *weights):
    n_things =X.shape[0]
    sum = np.zeros_like( X[0] )
    for ii in range(n_things):
        sum +=  X[ii] * weights[ii] 

    return sum

def _fit1(fit_sim,fit_data,p0):
    
    X = fit_sim.copy()
    Y = fit_data.copy()
    
    con, _  = curve_fit(_weighted_sum, X, Y, p0=p0,bounds=(0,np.inf))

    return con

# load sim and dists
print('load stuff')
cor_1gp2_confs =np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/1gp2_51sims_for_fit.npy')
exp_cpsi = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/exp_cpsi.npy')
sim_cpsi = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/sim_cpsi.npy')
all_dists = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/1gp2_51sims_Glu238_Arg90_Ca_dists.npy')
bins = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/hist_bins_51sims.npy')
n_shots = [36,72,108,144]
diff_hist=[]
for n_shot in n_shots:
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("num shots: %dk"%n_shot)
    # load the ave cors
    best_GDP_diff = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/ave_norm_cor_GDPdiff_phioffset30_nShots_%dk.npy'%n_shot)[10:]
    # best_GDP_err = np.load('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/ave_norm_cor_GDPdiff_err_phioffset30_nShots_%dk.npy'%n_shot)[10:]


    # normalize and interpolate
    print('normalizing and interpolating...')
    best_GDP_diff2, exp_cpsi2 = reorder_data(best_GDP_diff.copy(), exp_cpsi.copy())
    best_GDP_diff2 = (best_GDP_diff2+best_GDP_diff2[:,::-1])/2.

    cor_1gp2_confs = 0.5*(cor_1gp2_confs+cor_1gp2_confs[::-1])
    cor_1gp2_confs2 = np.zeros_like(cor_1gp2_confs)
    norm_cor_1gp2_confs = np.zeros_like(cor_1gp2_confs2)
        
    for ii in range(cor_1gp2_confs.shape[0]):
        cor_1gp2_confs2[ii],sim_cpsi2 = reorder_data(cor_1gp2_confs[ii].copy(), sim_cpsi.copy())
        for iq in range(best_GDP_diff2.shape[0]):
            norm_cor_1gp2_confs[ii,iq] = normalize(cor_1gp2_confs2[ii,iq])

    norm_best_GDP_diff = np.zeros_like(best_GDP_diff)
    for iq in range(best_GDP_diff2.shape[0]):
        norm_best_GDP_diff[iq] = normalize(best_GDP_diff2[iq])


    # interpolate data and simulations onto the same grid
    interp_num_phi =100
    new_cpsi = np.linspace(np.max( (exp_cpsi2.min(),sim_cpsi2.min()) )+0.05,
                          np.min((exp_cpsi2.max(), sim_cpsi2.max()))-0.05,
                          interp_num_phi,endpoint=False )

    interp_cor_1gp2_confs = np.zeros( (cor_1gp2_confs.shape[0],25,interp_num_phi) )
        
    for ii in range(cor_1gp2_confs.shape[0]):
        for iq in range(best_GDP_diff2.shape[0]):
            f = interp1d(sim_cpsi2[iq], norm_cor_1gp2_confs[ii,iq])

            interp_cor_1gp2_confs[ii,iq] = f(new_cpsi)

    interp_best_GDP_diff = np.zeros_like(interp_cor_1gp2_confs[0])

    for iq in range(best_GDP_diff2.shape[0]):
        f = interp1d(exp_cpsi2[iq],norm_best_GDP_diff[iq])
        interp_best_GDP_diff[iq] = f(new_cpsi)

    print('doing the fit...')
    # fit and compute x2
    num_model_1gp2=cor_1gp2_confs.shape[0]

    con_1gp2_best_GDP = np.zeros ( (25, num_model_1gp2), dtype = np.float32)

    x2_1gp2_best_GDP = np.zeros( 25 , dtype=np.float32)
    best_GDP_fits = np.zeros_like(interp_best_GDP_diff)

    for qidx in range(interp_best_GDP_diff.shape[0]):
       
        cons_best=_fit1(interp_cor_1gp2_confs[:,qidx], interp_best_GDP_diff[qidx], p0=[0.1]*num_model_1gp2)
        con_1gp2_best_GDP[qidx]=cons_best
        fit = _weighted_sum(interp_cor_1gp2_confs[:,qidx],*cons_best)
        resi_bes = interp_best_GDP_diff[qidx]-fit
        best_GDP_fits[qidx]=fit

        # compute x2 from the residuals
        x2_1gp2_best_GDP[qidx]= np.sqrt((resi_bes**2).mean())


    # get newly biased dist distribution
    normalized_weights = con_1gp2_best_GDP/con_1gp2_best_GDP.sum(-1)[:,None]
    hist=np.histogram(all_dists,bins=bins)
    labels=np.digitize(all_dists,hist[1][:-1])-1

    print labels.shape
    weight_bin_count=None
    for iq in range(con_1gp2_best_GDP.shape[0])[:]:
        if weight_bin_count is None:
            weight_bin_count=np.array([np.sum(normalized_weights[iq,labels==ll])*(hist[0][ll]) for ll in range(labels.max()+1)])
        else:
            weight_bin_count+=np.array([np.sum(normalized_weights[iq,labels==ll])*(hist[0][ll]) for ll in range(labels.max()+1)])

    diff_hist.append(weight_bin_count/weight_bin_count.sum()-(hist[0]/float( hist[0].sum() )))
diff_hist = np.array(diff_hist)
np.save('/reg/d/psdm/cxi/cxilp6715/results/running_ave_cors/51sims_runnig_ave_diff_histograms.npy', diff_hist)