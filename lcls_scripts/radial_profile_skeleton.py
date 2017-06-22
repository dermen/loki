"""
skeleton script for doing peak finding on radial profiles... 
to help with WAXS analysis
 -dermendarko@gmail.com
"""


import h5py
import numpy as np
from scipy.signal import argrelmax

from loki.RingData import RadialProfile
from loki.utils.postproc_helper import smooth


def fit_line(data):
    """
    fit a line to the end points of a 1-D numpy array

    data, 1D numpy array, e.g. a radial profile
    """
    left_mean = data[:10].mean()
    right_mean = data[-10:].mean()
    slope = (right_mean - left_mean) / data.shape[0]
    
    x = np.arange(data.shape[0])
    return slope * x + left_mean

def norm_data(data):
    """
    Normliaze the numpy array 
     
    data, 1-D numpy array
    """
    data_min = data.min()
    data2 = data- data.min()
    return data2/ data2.max()

def rad_profiles( img_gen, cent, img_mask, rad_range ):
    """
    Makes radial profiles for each image in a generator of images... 

    img_gen, generator of 2D numpy image arrays

    cent, where forward beam hits detector, in pixel units
        (fast_scan coordinate, slow_scan coorinate)
    
    img_mask, 2D boolean numpy array, same shape as the images
        0 is masked, 1 is not masked

    rad_range, tuple of (min radius, max radius) in pixel units,
        (radius range of the radial profile)

    """
    print("Computing radial profiles.. ")
    rp = RadialProfile( cent, mask=img_mask )
    rad_pros = [rp.calculate(i)[rad_range[0]:rad_range[1]] for i in img_gen ]
    return rad_pros




def main1():
    import os

#   debug save
    sname = "debug.npy"
    if os.path.exists(sname):
        return np.load(sname)

#   number of phi around the ring
    nphi = 4320 

#   minimium and maximum radii for calculating radial profile
    rmin = 700 # pixel units
    rmax = 950

#   point where forward beam intersects detector
    cent = np.array([ 1199.0093935 ,  1197.28156522])

#   pre-made mask file
    mask = np.load('/home/mender/mpccd_mask_dna.npy')

#   this is a sacla file on texas, we should change to small data format
    fname = '/media/sastower1/data/sacla_201504/Dataget/run296314/MapData/296314_1.h5' 
    f = h5py.File( fname, 'r')
    grp = f['run_296314/detector_2d_assembled_1'] # navigate to the shot tags
    tags = grp.keys()[1:] # get the shot tags, first key is not a tag at SACLA... 

#   make a generator of images cause there are a lot
    imgs = ( grp[ '%s/detector_data'%t ].value 
        for t in tags )

#~~~~~~~~~this is the radial profiler~~~~~~~~~~~~~~~
    rad_pros = rad_profiles( imgs, cent, mask, rad_range=(rmin, rmax) )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    np.save(sname, rad_pros)

    return rad_pros


def main2(rad_pros):
    """
    if we want we can peak select the profiles for looking at things like jitter
    some paraemters
    """

#~~~~~~ Below are the paraemters used in this main

#   smoothing
    beta = 50 # smoothing factor
    window_size = 200 # pixel units
    
#   maxima detection
    order = 10 # defines minimum neighborhood for local maxima (in radial pixel units)

#   paraemters for peak validation
    pk_range = (100, 230) # radial pixel units, relative to the range of the radial profiles
#   e.g. will ensure the detected peak lies on rad_pofile[ pk_range[0] : pk_range[1]] 


    import pylab as plt
    plt.plot( rad_pros[0])
    plt.suptitle("Looking for peaks bound by vertical lines...\n\
        change pk_range arg to adjust... ")
    plt.vlines( pk_range, rad_pros[0].min(), rad_pros[1].max())
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#   before peak fitting we smooth
    smooth_pros = [ smooth(r-fit_line(r), beta=beta, window_size=window_size) 
        for r in rad_pros]

#   we can normalize them if we feel like it... 
    norm_pros = map( norm_data, smooth_pros)

#   we can find local maxima in the smoothed normalzied radial profiles... 
    rel_max = [argrelmax(n, order=order)[0] for n in norm_pros ]

#   this should pick out the good hits...  but we will need to play around with it at beamtime
    
    hit_inds = []
    for i, mx in enumerate( rel_max):
#       make sure there is only one peak!
        if not len(mx) == 1:
            continue

#       make sure the peak lies in the desired range.. 
        pk_pos = mx[0]
        if not pk_range[0] < pk_pos < pk_range[1] : 
            continue

#       make sure the peak value is max in the original profile, 
#       because it was selected using line-subtracted profile
        pro = rad_pros[i]
        pk_val = pro[ pk_pos] 
        if pro[0] < pk_val and  pro[-1] < pk_val :
            hit_inds.append( i )  
    
    print("Picked out %d/%d hits!"%( len(hit_inds), len( rad_pros) ))

    return rad_pros, hit_inds 

if __name__ == "__main__":
#   if you just want radial profiles use 
    rad_pros = main1()

#   I know this is a hit... 
    import pylab as plt
    plt.plot( rad_pros[175] ) 
    plt.show()
    
#   or

#   if you want to play with the hit finder use
    rad_pros, hit_inds =  main2( main1() ) 
    if hit_inds:
        plt.plot( rad_pros[ hit_inds[0] ] ) 
        plt.show()
    else:
        print("Not hits, womp womp womp... ")
