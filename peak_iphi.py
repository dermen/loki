from pylab import *

import os
import h5py
os.chdir('/data/work/mender')
import postproc_helper

from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

#wavelen = 1.442 # angstrom
#k = 0.94 # factor spherical shape; cubic internal


#f = h5py.File('/data/sacla_gold_Feb2014/interped_178802.hdf5','r')
#p = f['polar_data']
#d = p[10][26]

def get_NP_sizes(Iphi, wavelen=1.442, q=2.668, k=0.94, rad='sphere'):
    """
    Iphi is an angular intensity profile with outlier Bragg peaks
    wavelen is photon wavelength in angstroms
    q is scattering vector in inverse angstrom
    k is a scherrer factor (default is 0.94 for spherical NPs with cubic internals
    rad determines which volume equation to use in determining NP width. use
        'tetra' for tetrahedral volume, default is'sphere' for spherical volume
    Returns the outlier NP size estimates in nanometers
    """
    assert( rad in ['sphere','tetra'] )
    # scherrer params
    th = arcsin( q * wavelen/ (4*pi) ) # bragg angle
    cth = cos(th)
    fact = k * wavelen / cth
    
    if rad == 'sphere':
        rad_form = lambda beta: fact* power(4*pi/3., 1/3.) / beta
    else:    
        rad_form = lambda beta: fact* power(6, 1/3.) * power(2, 1/6.) / beta

    d = Iphi

    N = d.shape[0] # 5000
    dphi = 2*pi / N

    d_ = postproc_helper.smooth( d,30,10)

    x = linspace( 0,N, 10*N )

    I = interp1d(arange(N), d_, bounds_error=0, 
            fill_value=median(d_) )

    dx = I(x)
    peak_pos = where( postproc_helper.is_outlier( I(x) ,3) )[0]
    d_peaks = ones_like( dx)*median( d_ )

    d_peaks[ peak_pos ] = dx[peak_pos ]
    edge = zeros_like( dx)
    edge[peak_pos] = 1

    RE = where( roll(edge,-1) - edge < 0 )[0]
    LE = where( roll(edge,1) - edge < 0 )[0]
    mins = argrelextrema( d_peaks, np.less )[0]
    all_mins =  sort(hstack( (LE, RE, mins)))
    maxs = argrelextrema( d_peaks, np.greater )[0]

    diams = []
    for i,mx in enumerate(maxs):
        minL = all_mins[ all_mins < mx]#[-1]
        minR = all_mins[ all_mins > mx]#[0]
        if minL.size ==0 or minR.size ==0:
            continue
        minL = minL[-1]
        minR = minR[0]
        a = arange(minL, minR)
        ydata = d_peaks[a]
        xdata = x[a]
        mu = x[mx]
        offset = median(d_)
        var = sqrt(10.)
        gfit = postproc_helper.fit_gauss_fixed_mu_fixed_off(ydata, xdata, mu, var, offset)
        if gfit is None:
            continue
        gvar = gfit[0][1]
        width = 2*sqrt(2*log(2)) * sqrt(gvar)
        beta = width*dphi
        radius = rad_form(beta)
        diam_nm = radius/5
        if isnan(diam_nm):
            continue
        diams.append(diam_nm)
    
    return diams

