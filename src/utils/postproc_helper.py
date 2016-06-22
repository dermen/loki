import sys
import os
import re

import json
import h5py
import pandas
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema


def is_outlier(points, thresh=3.5):
    """
    http://stackoverflow.com/a/22357811/2077270
    
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745*diff / med_abs_deviation

    return modified_z_score > thresh


def remove_peaks(Ring, Mask, thick=None, peak_thresh=3.0,
                    coef=None):
    """
    Ring - angular profile (1-d array I(phi) ) 
    Mask - angular profile mask (0 is masked, 1 is kept)
    thick - extent of the peak;  detected peak position +/- thick will 
            all be masked. Units of angular resolution around ring (2pi/Nphi)
    peak_thresh - threshhold for detecting outlier peaks
    coef - if true, then a polynoial is fitted using these coefficients 
                and subtacted prior to peak detection

    returns a 3-tuple
     1) data with detected peaks set to 0, 
     2) an updated mask to include detected peaks 
     3) the removed data with everything else set to 0
    """
    points = []
    points_map = []
    counter = 0

    ring = Ring.copy()
    mask = Mask.copy()

    nphi = ring.shape[0]

    xvals = [i for i in xrange(nphi) if mask[i]] 

    for i,p in enumerate(ring):
        if mask[i]:
            points.append(p) 
            points_map.append( [ counter, i ] )
            counter += 1

    points = np.array(points)
    points_map = dict(points_map)
    if coef is not None:
        poly = np.polynomial.chebyshev.chebval(xvals, coef)
        points -= poly
     
    outliers = is_outlier( points, peak_thresh )
    
    removed = np.zeros_like(ring )
    where_outliers = np.where(outliers)[0]
    if where_outliers.size:
        outliers_remapped = [points_map[o] for o in where_outliers ]
        if thick is not None:
            win = np.arange( -thick, thick+1)
            outliers_with_win = np.unique( [ x for ol in outliers_remapped
                                               for x in ol+win
                                               if x >= 0 and x < nphi])
            mask[outliers_with_win] = 0
            removed[ outliers_with_win] = ring[ outliers_with_win]
        else:
            mask[outliers_remapped] = 0
            removed[ outliers_remapped] = ring[outliers_remapped]
        
    return ring*mask, mask, removed

def get_ring( pdata, pmask, iq, del_q=2, rm_peaks=True, 
                rm_extent=1, rm_thresh=2.):
    """
    'pdata' - a 2d polar image shape is (N_radial, N_polar)
    'pmask' - a 2d polar mask same shape as pmask
    'iq'    - radial index of ring; e.g.  pdata[iq, :]  
    'del_q' - thickness of ring in pixek units such that 
                I(phi) = pdata[iq-del1:iq+delq+1,:].mean(0)
    'rm_peaks' - whether to remove peaks in the radial 
    'rm_extent' - how thick of a pk to remove (0 means just remove 
                the pk, 1 means remove the peak and its nearest 
                neighbors.. etc)
    'rm_thresh' - threshold for removing peaks
    """
    
    r = np.arange( iq-del_q, iq+del_q+1)
    r = r[ r < pdata.shape[0] ]
    r = r[r > 0] 

    mask = np.floor( pmask[r].mean(0))
    ring = mask*(pdata[r].mean(0))
    if rm_peaks:
        ring, mask,_ = remove_peaks( ring, mask, 
                            thick=rm_extent,
                            peak_thresh=rm_thresh)
    ring *= mask
    med = np.median( ring[ ring > 0 ] )
    return med, ring, mask


def fit_periodic(data, mask, deg, overlap=0.2):
    """
    fit a polynomial of degree deg to a periodic function
    that has a mask

    data - 1d array of periodic function (full interval) e.g. the 
        intensity from 0 -2PI at fixed scattering angle
    mask - masked values corresponding to data (0 is masked, 1 is kept)
    overlap - how far to extend data (fraction of data.shape[0]
    deg - degree of the polynomial fit to data
    """
    #assert np.all( data.shape == mask.shape)
    n = len(data)
    nn = int(overlap *n)

    data_m = data*mask
    y = np.zeros(n+2*nn)
    y[nn:n+nn] = data_m
    y[:nn] = data_m[-nn:]
    y[-nn:] = data_m[:nn]

    x = np.arange(-nn,n+nn)
    x_m = np.ma.masked_where( y==0, x)

    fit = np.ma.polyfit( x_m, y, deg=deg )
    yfit = np.polyval(fit, np.arange(n) )*mask
    ofit = np.polynomial.chebyshev.chebfit(x, 
                        np.polyval(fit,x), deg=deg)

    return ofit, fit, yfit


def fft_crosscorr( arx,ary):
    n = arx.shape[-1]
    axis = len(arx.shape)-1
    fx = np.fft.rfft( arx, n = n, axis=axis )
    fy = np.fft.rfft( ary, n = n, axis=axis )
    return np.fft.irfft( fx* np.conjugate( fy), n = n,axis=axis)


def random_pairs(total_elements, num_pairs):
    np.random.seed()
    inter_pairs = []
    factor = 2
    while len(inter_pairs) < num_pairs:
        rand_pairs   = np.random.randint( 0, 
                                        total_elements, 
                                        (num_pairs*factor,2) )
        unique_pairs = list( set( tuple(pair) 
                            for pair in rand_pairs ) )
        inter_pairs  = filter( lambda x:x[0] != x[1], 
                                unique_pairs)
        factor += 1
    return np.array(inter_pairs[0:num_pairs])


def smooth(x, beta=10.0, window_size=11):
    """
    Apply a Kaiser window smoothing convolution.
    
    Parameters
    ----------
    x : ndarray, float
        The array to smooth.
        
    Optional Parameters
    -------------------
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta 
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.
        
    Returns
    -------
    smoothed : ndarray, float
        A smoothed version of `x`.
    """
    
    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # apply the smoothing function
    s = np.r_[x[window_size-1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve( w/w.sum(), s, mode='valid' )
    
    # remove the extra array length convolve adds
    b = (window_size-1) / 2
    smoothed = y[b:len(y)-b]
    
    return smoothed


def lorentz( x,p0,p1,p2, p3) :
    return  (p2/np.pi) * (p1 / 2. ) * (1/( (x-p0)*(x-p0) + p1*p1/4. )) + p3

def lorentz2( x,p0,p1) :
    return  (1./np.pi) * (p1 / 2. ) * (1/( (x-p0)*(x-p0) + p1*p1/4. )) 

def lorentz2_offset( x,p0,p1, offset) :
    return  offset + (1./np.pi) * (p1 / 2. ) * (1/( (x-p0)*(x-p0) + p1*p1/4.)) 

def fit_lorentz_offset_fixed_mu( ydata, xdata, mu):#, gamma_i):
    amp = ydata.ptp() /2.
    offset  = ydata.min()
    gamma_i = 2 / np.pi/ amp 

    L = lambda x,p1,offset_param : offset_param + (1./np.pi)*(p1/2.)\
                *(1/((x-mu)*(x-mu)+p1*p1/4.))
    
    try:
        fit,success = optimize.curve_fit(L, xdata=xdata,
                                    ydata=ydata, p0=guess)
    except RuntimeError:
        return None
    
    return fit, success, gauss_offset(xdata, *fit)

def root( x, *p):
    m,b = p
    return m*np.power(x,1/2.) + b


def gauss_offset(x, *p):
    A, mu, var, offset = p
    return A*np.exp(-(x-mu)**2/(2.*var)) + offset

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_var(x, *p):
    A, mu, var = p
    return A*np.exp(-(x-mu)**2/(2.*var))


def fit_gauss_offset( peak_pro, xdata, width ):
    mu     = xdata[ peak_pro.argmax() ] 
    sig    = width
    amp    =  peak_pro.ptp() /2.
    offset   =  peak_pro.min()

    guess =( amp, mu,  sig**2, offset )

    try:
        fit,success = optimize.curve_fit( gauss_offset, 
                        xdata=xdata, ydata=peak_pro, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,gauss_offset(xdata, *fit)

def fit_gauss_offset_fixed( peak_pro, xdata, mu, sigma ):
    amp    =  peak_pro.ptp() /2.
    offset   =  peak_pro.min()

    resample_x = np.linspace( xdata.min(), xdata.max(), 1000 )
    peak_pro_I = interp1d( xdata, peak_pro )
    resample_peak = peak_pro_I(resample_x)

    guess =( amp, offset )

    Gauss = lambda x,amp,offset: amp*np.exp(-(x-mu)**2/(2.*sigma**2)) + offset

    try:
        fit,success = optimize.curve_fit( Gauss, 
                        xdata=resample_x, ydata=resample_peak, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,Gauss(xdata, *fit)

def fit_gauss_offset_fixed_mu( peak_pro, xdata, mu, var_guess ):
    amp    =  peak_pro.ptp() /2.
    offset  =  peak_pro.min()

    resample_x = np.linspace( xdata.min(), xdata.max(), 1000 )
    peak_pro_I = interp1d( xdata, peak_pro )
    resample_peak = peak_pro_I(resample_x)

    guess =( amp, offset, var_guess )

    Gauss = lambda x,amp, offset, var: amp*np.exp(-(x-mu)**2/(2.*var)) + offset

    try:
        fit,success = optimize.curve_fit( Gauss, 
                        xdata=resample_x, ydata=resample_peak, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,Gauss(xdata, *fit)

def fit_gauss_fixed_mu_fixed_off( peak_pro, xdata, mu, var_guess, offset ):
    amp    =  peak_pro.max() - offset

    #resample_x = np.linspace( xdata.min(), xdata.max(), 1000 )
    #peak_pro_I = interp1d( xdata, peak_pro )
    #resample_peak = peak_pro_I(resample_x)

    guess =( amp, var_guess )

    Gauss = lambda x,amp, var: amp*np.exp(-(x-mu)**2/(2.*var)) + offset

    try:
        fit,success = optimize.curve_fit( Gauss, 
                        xdata=xdata, ydata=peak_pro, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,Gauss(xdata, *fit)

def fit_gauss_var( peak_pro, xdata, var ):
    mu     = xdata[ peak_pro.argmax() ] 
    var    = var
    amp    =  peak_pro.ptp() /2.

    guess =( amp, mu,  var )

    try:
        fit,success = optimize.curve_fit( gauss_var, 
                        xdata=xdata, ydata=peak_pro, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,gauss(xdata, *fit)

def fit_gauss( peak_pro, xdata, width ):
    mu     = xdata[ peak_pro.argmax() ] 
    sig    = width
    amp    =  peak_pro.ptp() /2.
    offset   =  peak_pro.min()

    guess =( amp, mu,  sig )

    try:
        fit,success = optimize.curve_fit( gauss, 
                        xdata=xdata, ydata=peak_pro, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,gauss(xdata, *fit)

def fwhm_gauss( peak_pro, xdata, width):
    a,b,c = fit_gauss( peak_pro, xdata, width)
    fwhm = 2.3548 * a[2]
    return fwhm

def amp_gauss( peak_pro, xdata, width):
    a,b,c = fit_gauss( peak_pro, xdata, width)
    amp = a[0]
    return amp

def mu_gauss( peak_pro, xdata, width):
    a,b,c = fit_gauss( peak_pro, xdata, width)
    mu = a[1]
    return mu

def delta_pix(qR, NP_diam, wavelen, q111, fraction=1/3. ):
    NP_vol = (4/3)*np.pi *( (NP_diam / 2 )**3)
    NP_size = np.power( NP_vol, 1/3.) * fraction

    th = np.arcsin(q111 * wavelen / 4 / pi)
    N_pix = 4*np.pi *0.9 * (qR /q111) /  (NP_size * np.cos(th)) 
    return N_pix

def scherrer_pixel( qR, pixsize, detdist, wavelen, NP_diam, extra_fac=1., K=.9 ):
    NP_vol = (4/3)*np.pi *( (NP_diam / 2 )**3)
    NP_size = np.power( NP_vol, 1/3.)
   
    th = np.arctan(qR*pixsize / detdist)/2.
    del_2th = K * wavelen / NP_size / np.cos(th) / extra_fac
    pix_per_del_2th = qR*del_2th
    return pix_per_del_2th



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
    th = np.arcsin( q * wavelen/ (4*np.pi) ) # bragg angle
    cth = np.cos(th)
    fact = k * wavelen / cth
    
    if rad == 'sphere':
        rad_form = lambda beta: fact* np.power(4*np.pi/3., 1/3.) / beta
    else:    
        rad_form = lambda beta: fact* np.power(6, 1/3.) * np.power(2, 1/6.) / beta

    d = Iphi

    N = d.shape[0] # 5000
    dphi = 2*np.pi / N

    d_ = smooth( d,30,10)

    x = np.linspace( 0,N, 10*N )

    I = interp1d(np.arange(N), d_, bounds_error=0, 
            fill_value=median(d_) )

    dx = I(x)
    peak_pos = np.where( is_outlier( I(x) ,3) )[0]
    d_peaks = np.ones_like( dx)*np.median( d_ )

    d_peaks[ peak_pos ] = dx[peak_pos ]
    edge = np.zeros_like( dx)
    edge[peak_pos] = 1

    RE = np.where( np.roll(edge,-1) - edge < 0 )[0]
    LE = np.where( np.roll(edge,1) - edge < 0 )[0]
    mins = argrelextrema( d_peaks, np.less )[0]
    all_mins = np.sort(hstack( (LE, RE, mins)))
    maxs = argrelextrema( d_peaks, np.greater )[0]

    diams = []
    for i,mx in enumerate(maxs):
        minL = all_mins[ all_mins < mx]#[-1]
        minR = all_mins[ all_mins > mx]#[0]
        if minL.size ==0 or minR.size ==0:
            continue
        minL = minL[-1]
        minR = minR[0]
        a = np.arange(minL, minR)
        ydata = d_peaks[a]
        xdata = x[a]
        mu = x[mx]
        offset = np.median(d_)
        var = np.sqrt(10.)
        gfit = fit_gauss_fixed_mu_fixed_off(ydata, xdata, mu, var, offset)
        if gfit is None:
            continue
        gvar = gfit[0][1]
        width = 2*np.sqrt(2*log(2)) * np.sqrt(gvar)
        beta = width*dphi
        radius = rad_form(beta)
        diam_nm = radius/5
        if np.isnan(diam_nm):
            continue
        diams.append(diam_nm)
    
    return diams


