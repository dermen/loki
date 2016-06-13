from pylab import * 
from scipy.interpolate import RectBivariateSpline
from scipy import odr
from scipy.ndimage import zoom

class RingFit:
    def __init__( self, img ):
        '''
        Initialize a class for fitting shapes to images with shapes on them
        ====================================================================
        img - 2d float np.array
        '''

#       Equation for an ellipse
        f_ellipse = lambda beta,x : (1/beta[2]/beta[2])*( x[0] -beta[0])**2 \
                                  + (1/beta[3]/beta[3])*( x[1] -beta[1])**2-1
#       Equation for a circle        
        f_circle  = lambda beta,x : ( x[0] - beta[0] )**2 \
                                  + ( x[1] - beta[1])**2 - beta[2]**2
#       Model classes
        self.ellipse_model = odr.Model( f_ellipse, implicit=True)
        self.circle_model  = odr.Model( f_circle, implicit=True)

#       make 2d and 1d index arrays       
        self.y,self.x = np.indices( img.shape )
        self.x1   = self.x.ravel()
        self.y1   = self.y.ravel()
        
        self.img = np.copy(img)

    def _remove_highest( self, data_array, num_high_pix):
#       removes the highest pixels so they don't corrupt the fit 
#           (e.g. bad pixels)
        high_inds = np.argsort( data_array )[:num_high_pix]
        data_array[ high_inds ] = 0
        return data_array

    def fit_circle( self, beta_i, num_fitting_pts=5000, 
                    ring_width=20 , num_high_pix=20  ):
        '''
        Fit a circle to a ring image
        ============================== 
        beta_i          - float tuple , ( x_center, y_center,radius )
        num_fitting_pts - int, number of pixels to include in fit 
        ring_width      - int, width ring-like region on image that contains 
                                ring of interest (pixels units)
        num_high_pix    - int, remove this many pixels before running fit 
                            (removes artificial high pixels that might
                            bias the fit)
        '''
#       radius of each pixel 
        r = sqrt ( (self.x - beta_i[0])**2 + (self.y - beta_i[1])**2  )
        img_copy = np.copy ( self.img ) 
#       remove all pixels outside of a radius range        
        img_copy[ r > beta_i[2] + ring_width/2 ] = 0
        img_copy[ r < beta_i[2] - ring_width/2 ] = 0

#       make the image 1-D
        img1 = img_copy.ravel()
        self._remove_highest( img1 , num_high_pix )

        num_pts = np.where( img1 > 0 )[0].shape[0]
        if num_fitting_pts >= num_pts :
            num_fitting_pts = num_pts / 2

#       find indices of remaining pixels in order of decreasing intensity 
        inds = np.argsort( img1 )[::-1][: num_fitting_pts]

#       use odr module to fit data to circle model
        pts = np.row_stack( [self.x1[ inds], self.y1[ inds]]   )
        lsc_data = odr.Data( pts , y=1)
        lsc_odr = odr.ODR( lsc_data, self.circle_model, beta_i)
        lsc_out = lsc_odr.run()
        return lsc_out.beta

    def fit_ellipse(self, beta_i  ,num_fitting_pts=5000, 
                        ring_width=40,num_high_pix = 20):
        '''
        Fit an ellipse to a ring image
        ============================== 
        beta_i          - float tuple , (x_center, y_center, x_radius,y_radius)

        num_fitting_pts - int, number of pixels to include in fit 
        
        ring_width      - int, width ring-like region on image that contains 
                                ring of interest (pixels units)
        
        num_high_pix    - int, remove this many pixels before running fit 
                                (removes artificial high pixels that might
                                bias the fit)
        '''
        
        r = np.sqrt ( (self.x - beta_i[0])**2 + (self.y - beta_i[1])**2  )
        self.img[ r > beta_i[2] + ring_width/2 ] = 0
        self.img[ r < beta_i[2] - ring_width/2 ] = 0

        img1 = self.img.ravel()
        self._remove_highest( img1 , num_high_pix )

        num_pts = np.where( img1 > 0 )[0].shape[0]
        if num_fitting_pts >= num_pts :
            num_fitting_pts = num_pts / 2

        inds = np.argsort( img1 )[::-1][: num_fitting_pts]

        pts = np.row_stack( [ self.x1[ inds], self.y1[ inds]  ] )
        lsc_data = odr.Data( pts, y=1)
        lsc_odr = odr.ODR( lsc_data, self.ellipse_model, beta_i)
        lsc_out = lsc_odr.run()
        return lsc_out.beta

# ideally we would only need to compute the polar mask once per exposure...
# but this code framework makes that difficult.
class RingFetch:
    def __init__(self, a, b, img, mask=None, q_resolution=0.05, phi_resolution=0.5,
                    wavelen=1.41, pixsize=0.00005, detdist=0.051):
        '''
        Description
        ===========
        Sums the intensities in a diffraction image to form effective pixel
        readouts at a specified resolution.

        Parameters
        ==========

        `a`, `b` is the horizontal, vertical pixel coordinate where
            forward beam intersects detector
        
        `img` is a two-dimensional diffraction image with ring patterns

        `mask` is a boolean array where False,True represent masked,
            unmasked pixels in an image.
        
        `q_resolution`, `phi_resolution` are desired resolutions of
            one-dimensional output ring in radial, azimuthal 
            dimensions (inverse Angstrom and degree units, respectively)
        
        `wavelen`, `pixsize`, `detdist` are photon-wavelength, pixel-size 
            (assuming square pixels) and sample-to-detector-distance in 
            Angstrom, meter, and meter units, respectively.
        '''
        self.a = a
        self.b = b
        self.img = img
        
        if mask is not None:
            assert(mask.shape == self.img.shape)
            self.mask = mask
        
        possible_max_radii = (self.img.shape[1] - self.a, 
                              self.a , 
                              self.img.shape[0]-self.b, 
                              self.b)
        self.maximum_allowable_ring_radius = np.min( possible_max_radii)
        
        self.q_resolution = q_resolution
        self.phi_resolution = phi_resolution
        self.num_phi_nodes =  int ( 360.  / self.phi_resolution)

#       convert pixel radius to q
        self.r2q = lambda R: np.sin( np.arctan(R*pixsize/detdist)/2.) \
                                *4*np.pi/wavelen
#       ... and vice versa
        self.q2r = lambda Q: np.tan( 2*np.arcsin(Q*wavelen/4/np.pi)) \
                                *detdist/pixsize
    

    def fetch_a_ring(self, ring_radius, poly_deg=10):
        '''
        Parameters
        ==========
        
        `ring_radius` is radius of ring in pixel units

        `poly_deg` is degree of polynomial to fit for estimating
                    the noise level around the ring (necessary
                    because the ring is typicalle anisotropic
                    due to shadows and shit!)
        
        Returns
        =======
       
        A tuple of ( `phi_values`, `polar_ring_final`)

        `phi_values` are the azimuthal centers to the effective pixels 
            at the desired resolution

        `polar_ring_final` is the corresponding intensities in the 
            effective pixels
        '''
        if self.img is None:
            raise( 'Set the working image first!' )
        
        self.poly_deg = poly_deg

#       ########################
#       get the radial bin count
#       ########################
        q_of_ring = self.r2q(ring_radius)

#       min q for ring at desired resolution
        qmin = q_of_ring - self.q_resolution/2. 

#       max q for ring at desired resolution
        qmax = q_of_ring + self.q_resolution/2.

#       qmin/qmax in radial pixle units
        rmin = int( self.q2r(qmin) ) 
        rmax = int(np.ceil( self.q2r(qmax) )) # "" ""

        assert( rmax < self.maximum_allowable_ring_radius)

        nphi_min = int( 2 * np.pi * rmin )
        assert(nphi_min > self.num_phi_nodes)

        print ( 'rmin  / rmax = %d/%d'%(rmin,rmax) )

#       number of radial pixel units across ring
        pix_per_delta_q = rmax-rmin

#       store the output
        polar_ring_final = np.zeros((pix_per_delta_q+1, self.num_phi_nodes ) )
        azimuthal_values = np.zeros((pix_per_delta_q+1, self.num_phi_nodes ) )


        for radius_index, radius in enumerate( xrange(rmin, rmax+1)):
#           number of azimuthal points in radial image
            nphi = int( 2*np.pi*radius) 


#           ########################################################
#           Here is where the magic begins
#           ########################################################
#           make the RingData interpolator object
#           (The trick here is that we interpolate the image at single
#           pixel precision (hence nphi = 2*pi*radius) and we only
#           inteprolate a ring that is 1 pixel wide at given radius.
#           In this way we follow the resolution of the detector 
            InterpSimp = InterpSimple( self.a, self.b, radius+1,
                            radius, nphi, raw_img_shape=self.img.shape)

#           now interpolate that image
            polar_img = InterpSimp.nearest(self.img)

            self.polar_ring = polar_img[0]
            
#           #########################################
#           Fill in masked gaps with a Gaussian noise
#           #########################################
            if self.mask is not None:
#               interpolate the polar mask
                polar_mask = InterpSimp.nearest(self.mask.astype(int))
                self.polar_ring_mask = polar_mask[0]
                self._fill_polar_ring( nphi )

#           now that we filled in the gaps, let's do the azimuthal binning
            phi_nodes = np.linspace(0, nphi, self.num_phi_nodes+1,
                                        endpoint=1 )
            phi_nodes = phi_nodes.astype(int)

            phi_node_index =np.array( [ int(.5*phi_nodes[i] + .5*phi_nodes[i+1]) 
                                    for i in xrange(self.num_phi_nodes)])
            
#           sum the bins up
            pix_per_delta_phi = int( self.phi_resolution * ( nphi / 360. ) )
           
#           make the azimuthal bin edges 
#           (code vommit)
            phi_ranges = [ (i -pix_per_delta_phi/2, \
                            1*(pix_per_delta_phi%2) +i+pix_per_delta_phi/2 ) 
                            for i in phi_node_index ]

#           store the phi values of each node 
#           in case they are changing from radius to radius
            azimuthal_values[radius_index] = np.arange(nphi)[phi_node_index] \
                                                        *2*pi/nphi
#           sum up the intensities in each azimuthal bin bin
            polar_ring_final[radius_index] =  \
                                array( [ sum(self.polar_ring[ r1:r2 ]) 
                                        for r1,r2 in phi_ranges ] ) 
            
        return azimuthal_values.mean(0), polar_ring_final.sum(0)

    def _fill_polar_ring(self, nphi ):
#       (I am not sure but this may be important step since i think we want
#       the sum of the photons in a effective pixel, maybe for cross-corrs
#       this matters)
        xvals = np.arange( nphi)  # for polyfitting
        yvals = self.polar_ring* self.polar_ring_mask # for polyfitting
#           fit to the non-masked points
        poly_coef = np.polyfit( xvals[yvals > 0], yvals[yvals >0],
                                    deg=self.poly_deg )
#       find the polynomial fit across the range
        moving_mean = np.polyval( poly_coef, xvals )
#       isolate the moving mean within the gaps
        
        gapped_moving_mean = moving_mean[ yvals == 0 ]
#       find the standard deviation of the signal about the moving mean
        width = np.std((yvals-moving_mean)[ yvals > 0])
#       fill in the gaps wih a Gaussian noise of corresponding width
        self.polar_ring[ self.polar_ring_mask==0] = np.random.normal( \
                                                        gapped_moving_mean, 
                                                                width )


class InterpSimple:
    def __init__ ( self,  a,b, qmax,qmin, nphi, raw_img_shape, bin_fac=None ) : 
        '''
        Define a polar image with dimensions: (qmax-qmin) x nphi
        ==========================================================
        a,b           - float, x_center,y_center of cartesian image that is 
                                being interpolated
        
        qmin,qmax     - cartesianimage will be interpolated from 
                            q-qmin, q + qmax
        
        nphi          - int, azimuthal dimension of polar image 
                            (number of azimuthal points along polar image)
        
        raw_img_shape - int tuple, ydim, xdim of the raw image 
        
        bin_fac       - float, reduce the size of the cartesian image 
                            by this amount

        '''
        self.a = a      
        self.b = b     
        self.qmin = qmin
        self.qmax = qmax  
        self.raw_shape = raw_img_shape   
        self.phis_ring     = arange( nphi ) * 2 * pi / nphi
        self.num_phis_ring = nphi

        self.bin_fac = bin_fac
        
        if bin_fac:
            self.Y,self.X = zoom( np.random.random(raw_img_shape) , 
                                    1. / bin_fac, order=1 ).shape 
        else: 
            self.X = raw_img_shape[1] # fast dimension
            self.Y = raw_img_shape[0] # slow dimension

        
        self.Q   = vstack( [ ones(self.num_phis_ring)*iq 
                            for iq in arange( self.qmin,self.qmax ) ] )
        self.PHI = vstack( [ self.phis_ring 
                            for iq in arange( self.qmin,  self.qmax ) ] )
        self.xring = self.Q*cos(self.PHI-pi) + self.a
        self.yring = self.Q*sin(self.PHI-pi) + self.b

        self.xring_near = self.xring.astype(int) + \
                            np.round( self.xring - \
                            floor(self.xring) ).astype(int)
        self.yring_near = self.yring.astype(int) + \
                            np.round( self.yring - \
                            floor(self.yring) ).astype(int)
#       'C' style ordering
        self.indices_1d = self.X* self.yring_near + self.xring_near 

    def nearest( self, data_img , dtype=np.float32):
        '''return a 2d np.array polar image (fastest method)'''
        if self.bin_fac :
            data_img = zoom( data_img, 1. / self.bin_fac, order=1 ) 
        data = data_img.ravel()
        return data[ self.indices_1d ]


class RadialProfile:
    def __init__( self, center, img_shape=(2399,2399), mask=None, 
                                minlength=1800 ):
        """
        center: float tuple, the usual (horizontal,vertical) tuple, 
                corresponding to the pixel coordinate where the forward 
                beam would pass throughthe area detector
        
        img_shape: integer typle, (Xdimension/fast, Ydimension/slow) for the 
                    images which will be radially binned.

        mask:   a 2D boolean array that has shape img_shape. True is non-masked
                and False is masked

        minlength:  The minimum length of the radial profile. Force this to be
                    high so that multiple radial profiles will have
                    the same length

        """
        self.x_center = center[0]
        self.y_center = center[1]
        self.minlength = minlength
        self.mask = mask

#       Make the radius of each pixel
        self.Y, self.X = np.indices( img_shape )
        self._set_R()
        self._set_normalization()

    def _set_R(self):
        self.R = np.sqrt((self.Y-self.y_center)**2 + \
                        (self.X-self.x_center)**2)
        self.R = self.R.astype(int)
        
    def _set_normalization(self):
        if self.mask is None:
            self.num_pixels_per_radial_bin = np.bincount( self.R.ravel(), 
                                    minlength=self.minlength)
        else:
            self.num_pixels_per_radial_bin = \
                            np.bincount(self.R[self.mask].ravel(), 
                                    minlength=self.minlength)

    def calculate(self, img):
        """
        img:   2-dimensional image to be radially binned
        
        returns the 1-dimensional radial profile of img
        """
        if self.mask is None:
            summed_intensity_per_radial_bin = np.bincount( self.R.ravel(), 
                                                weights=img.ravel(), 
                                                minlength=self.minlength)
        else:
            summed_intensity_per_radial_bin = np.bincount( self.R.ravel(), 
                                                weights=(self.mask*img).ravel(), 
                                                minlength=self.minlength)

        radial_profile = summed_intensity_per_radial_bin / \
                            self.num_pixels_per_radial_bin

        return np.nan_to_num(radial_profile)

    def update_center( self, new_center):
        self.x_center = new_center[0]
        self.y_center = new_center[1]
        self._set_R()
        self._set_normalization()


class DiffCorr:
    def __init__( self, shots ,delta_shot=None,pre_dif = True ):
        '''
        Initialize a correlation class for doing difference correlations 
        ================================================================
        shots       - 2d or 3d float np.array , shape is 
                        (num_shot x num_q x num_phi) or   
                        (num_shot x num_phi) .  
        delta_shot  - set the spacing between "consecutive shots" . 
                        A consecutive shot pair will be correlated
                        using the difference correlation method
        '''
        
        if pre_dif:
            self.shotsAB = shots
        else:
            shotsA = shots[0:-delta_shot]
            shotsB = shots[delta_shot:]
            self.shotsAB = shotsA - shotsB
    
    def _fft_autocorr(self, ar):
        n = ar.shape[-1]
        axis = len( ar.shape) -1
        fx = np.fft.rfft( ar, n = n, axis=axis )
        return np.fft.irfft( fx* conjugate( fx), n = n,axis=axis)

    def autocorr(self): #,num_high=0,num_low=0):
        '''
        Return the difference autocorrelation
        =====================================
        num_high/num_low - int number of high/low pixels to remove 
        before correlating
        '''
        self.corr = array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return self.corr
    

#~%^%^~!%^~!%~^!%&~!%^~!~&^!*~^!&*~^!&*~!^&%~!%$~&!%*&~*!^(~&!%&^~$!%&~$!%~


########################################################

# The following are out-dated and will be removed soon #

########################################################

def radial_profile( img, center=None, R = None, norm=None, 
                    mask=None, minlength = None) :
    '''
    Return the radial profile of an image
    ======================================
    center    - float tuple, (fast axis center, slow slow center )
    R         - float np.array ,  radial coordinate of each pixel from center 
    norm      - float np.array, number of pixels in each radial bin 
                    (needed if mask !- None)
    mask      - bool np.array, same shape as image, 1 is unmasked, 0 is masked
    minlength - int, minimum number of radial points
    '''
    if R == None:
        if mask == None:
            Y,X = indices( img.shape )
            R = sqrt( (X-center[0])**2 + (Y-center[1])**2 )
            R = R.astype(np.int)
            tbin = bincount( R.ravel(), img.ravel(), minlength = minlength )
            nr = bincount( R.ravel(), minlength = minlength )
            radial_profile = tbin / nr
        else:
            if norm == None:
                raise ("Please enter a normalization array for the mask provided")
            Y,X = indices( img.shape )
            R = sqrt( (X-center[0])**2 + (Y-center[1])**2 )
            R = R.astype(np.int)
            tbin = bincount( R.ravel(), (img*mask).ravel(), 
                        minlength = minlength )
            radial_profile = tbin / norm
    else:
        if mask == None:
            tbin = bincount( R.ravel(), img.ravel(), minlength = minlength )
            nr = bincount( R.ravel(), minlength = minlength )
            radial_profile = tbin / nr
        else:
            tbin = bincount( R.ravel(), (img*mask).ravel(), 
                minlength = minlength )
            radial_profile = tbin / norm
    
    return nan_to_num( radial_profile )

