import numpy as np
import h5py

def ave_all_cor(files, dataset = 'ave_norm_corr',
               phi_offset = 0):
    """
    Average quantities across different h5 files, weighted by the number of shots per file

    files - list of str,paths to the files containing quantities to average
    dataset - str, name of the data set to average in the h5 files
    phi_offset - int, number of pixels in phi to ignore on the two extremes of the correlations
    Ignoring these pixels means we are not looking at the very high values near 0 and pi

    """
    n_shots = []
    corrs = []
    for ff in files:
        h5 = h5py.File(ff,'r')
        n_shots.append(h5['num_shots'].value)

        num_phi = h5[dataset].shape[-1]

        try:
            corrs.append(h5[dataset].value[:,phi_offset:num_phi/2-phi_offset])
        except KeyError:
            continue
    corrs = np.array(corrs)
    n_shots = np.array(n_shots,dtype=float)
    
    num_shots = n_shots.sum().astype(float)
    n_shots/=num_shots
    
    ave_corr = (corrs * n_shots[:,None,None]).sum(0)
    return ave_corr, num_shots