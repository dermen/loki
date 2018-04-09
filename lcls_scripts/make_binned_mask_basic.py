

import numpy as np


from loki.RingData import RadialProfile, InterpSimple
from loki.utils.postproc_helper import smooth, bin_ndarray

img_sh = (1734, 1731)

cent_fname = '/reg/d/psdm/cxi/cxilp6715/results/shared_files/center.npy'
mask_fname = '/reg/d/psdm/cxi/cxilp6715/results/shared_files/mask_rough3.npy'
cent = np.load( cent_fname)
mask = np.load( mask_fname) 

#~~~~~ WAXS parameters
# minimium and maximum radii for calculating radial profile
waxs_rmin = 100 # pixel units
waxs_rmax = 1110
rp = RadialProfile( cent, img_sh, mask=mask  )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~ hit findin' parameters... 

beta = 50 # smoothing factor
window_size = 200 # pixel units

# maxima detection
order = 250 # defines minimum neighborhood for local maxima (in radial pixel units)
# paraemters for peak validation
pk_range = (800, 1045) # radial pixel units, relative to the range of the radial profiles
# e.g. will ensure the detected peak lies on rad_pofile[ pk_range[0] : pk_range[1]] 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~ interp parameters

# min ring radii
# desired dimension of image, these will be approximate
rbins =35
phibins = 360

interp_rmin = 100
interp_rmax = 450

rbin_fct = np.floor( (interp_rmax - interp_rmin) / rbins)
# adjust so our edge is a multiple of rbin factor
interp_rmax = int( interp_rmin + np.ceil( (interp_rmax - interp_rmin) / rbin_fct)*rbin_fct )

nphi = int( 2 * np.pi * interp_rmax )
phibin_fct = np.ceil( nphi / float( phibins ) )
nphi = int( np.ceil( 2 * np.pi * interp_rmax/phibin_fct)*phibin_fct) # number of azimuthal samples per bin

rbin_new = (interp_rmax- interp_rmin ) / rbin_fct
phibin_new = nphi / phibin_fct 
binned_pol_img_sh = ( int(rbin_new), int(phibin_new) )
print("polar image dimensions:  %d x %d"%(rbin_new, phibin_new))

Interp = InterpSimple( cent[0], cent[1] , interp_rmax, interp_rmin, nphi, img_sh)  
pmask = Interp.nearest(mask).astype(int).astype(float)
pmask_bn = bin_ndarray( pmask, binned_pol_img_sh)
pmask_bn = pmask_bn.astype(int)
pmask_bn = np.array(pmask_bn==pmask_bn.max(), dtype = int)

#np.save('/reg/d/psdm/cxi/cxilp6715/scratch/water_data/binned_pmask_basic.npy', pmask_bn)
np.save('/reg/d/psdm/cxi/cxilp6715/results/shared_files/binned_pmask_basic3.npy', pmask_bn)
