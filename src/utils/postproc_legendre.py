from numpy.polynomial.legendre import legfit, legval, leggauss
from scipy.interpolate import splev,splint,splrep

import numpy as np

def interpolate_corr( xold, xnew, signal ):
    
    # compute the bspline representation of the signal
    # k =5 gives the highest order

    tck = splrep(xold,signal,k=5,s=0)
    signal_interp = splev(xnew,tck, der=0)

    return signal_interp

def leg_proj_legguass(x, signal, weights,lmax):
    """projection into legendre polynomial using gauss-legendre polynomial
    """
    coefs = np.zeros(lmax + 1)
    weights_sum = np.sum(weights)

    for l in range(lmax + 1):
        if l%2==0:
            cc = np.zeros(l+1)
            cc[l] = 1
            coefs[l] = np.sum( weights*legval(x,cc)*signal) / weights_sum*(2*l+1)

    return coefs


def compute_legendre_projection(corr, cospsi, lmax, remove_zero=True):
    # define gaussian quadrature, the points between -1, 1 and the weights
    xnew, ws = leggauss( cospsi.shape[-1] )
    n_q = corr.shape[0]
    
    # interpolate
    cl = np.zeros( (  n_q, lmax + 1) )
    for i in range(n_q):
        signal =  corr[i,:].copy()
        signal_interp =  interpolate_corr( cospsi[i,:],
                                          xnew, signal)
        c =  leg_proj_legguass( xnew, signal_interp, ws, lmax)
        cl[i,:] = c
        
    if remove_zero:
        cl[:,0] = 0.
    return cl


        
def get_cpsi(num_phi, wavlen, q1, q2):
    """
    Returns cos psi values
    ================================================================
    num_phi  - int, number of phi values in the ring

    wavlen   - float, wavelength in angstrom

    q1, q2   - float, magnitudes of q values

    Returns:
    cpsi     - 1d array, shape = (num_phi)
    """
    phis = np.arange( num_phi ) * 2 * np.pi / num_phi
    th1 = np.arcsin( q1 * wavlen / 4. / np.pi )
    th2 = np.arcsin( q2 * wavlen / 4. / np.pi )
    cpsi = np.cos(phis) * np.cos(th1)*np.cos(th2) + np.sin(th1)*np.sin(th2)

    return cpsi