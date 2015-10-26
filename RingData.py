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
        f_ellipse = lambda beta,x : (1/beta[2]/beta[2])*( x[0] - beta[0] )**2 \
                                  + (1/beta[3]/beta[3])*( x[1] - beta[1]) **2 - 1
#       Equation for a circle        
        f_circle  = lambda beta,x : ( x[0] - beta[0] )**2 \
                                  + ( x[1] - beta[1]) **2 - beta[2]**2
#       Model classes
        self.ellipse_model = odr.Model( f_ellipse, implicit=True)
        self.circle_model  = odr.Model( f_circle, implicit=True)

#       make 2d and 1d index arrays       
        self.y,self.x = indices( img.shape )
        self.x1   = self.x.ravel()
        self.y1   = self.y.ravel()
        
        self.img = copy(img)

    def _remove_highest( self, data_array, num_high_pix):
#       removes the highest pixels so they don't corrupt the fit (e.g. bad pixels)
        high_inds = argsort( data_array )[:num_high_pix]
        data_array[ high_inds ] = 0
        return data_array

    def fit_circle( self, beta_i, num_fitting_pts=5000, ring_width=20 , num_high_pix=20  ):
        '''
        Fit a circle to a ring image
        ============================== 
        beta_i          - float tuple , ( x_center, y_center,radius )
        num_fitting_pts - int, number of pixels to include in fit 
        ring_width      - int, width ring-like region on image that contains ring of interest (pixels units)
        num_high_pix    - int, remove this many pixels before running fit (removes artificial high pixels that might bias the fit)
        '''
#       radius of each pixel 
        r = sqrt ( (self.x - beta_i[0])**2 + (self.y - beta_i[1])**2  )
        img_copy = copy ( self.img ) 
#       remove all pixels outside of a radius range        
        img_copy[ r > beta_i[2] + ring_width/2 ] = 0
        img_copy[ r < beta_i[2] - ring_width/2 ] = 0

#       make the image 1-D
        img1 = img_copy.ravel()
        self._remove_highest( img1 , num_high_pix )

        num_pts = where( img1 > 0 )[0].shape[0]
        if num_fitting_pts >= num_pts :
            num_fitting_pts = num_pts / 2

#       find indices of remaining pixels in order of decreasing intensity 
        inds = argsort( img1 )[::-1][: num_fitting_pts]

#       use odr module to fit data to circle model
        pts = row_stack( [self.x1[ inds], self.y1[ inds]]   )
        lsc_data = odr.Data( pts , y=1)
        lsc_odr = odr.ODR( lsc_data, self.circle_model, beta_i)
        lsc_out = lsc_odr.run()
        return lsc_out.beta

    def fit_ellipse(self, beta_i  ,num_fitting_pts=5000, ring_width=40,num_high_pix = 20):
        '''
        Fit an ellipse to a ring image
        ============================== 
        beta_i          - float tuple , ( x_center, y_center, x_radius, y_radius )
        num_fitting_pts - int, number of pixels to include in fit 
        ring_width      - int, width ring-like region on image that contains ring of interest (pixels units)
        num_high_pix    - int, remove this many pixels before running fit (removes artificial high pixels that might bias the fit)
        '''
        
        r = sqrt ( (self.x - beta_i[0])**2 + (self.y - beta_i[1])**2  )
        self.img[ r > beta_i[2] + ring_width/2 ] = 0
        self.img[ r < beta_i[2] - ring_width/2 ] = 0

        img1 = self.img.ravel()
        self._remove_highest( img1 , num_high_pix )

        num_pts = where( img1 > 0 )[0].shape[0]
        if num_fitting_pts >= num_pts :
            num_fitting_pts = num_pts / 2

        inds = argsort( img1 )[::-1][: num_fitting_pts]

        pts = row_stack( [ self.x1[ inds], self.y1[ inds]  ] )
        lsc_data = odr.Data( pts, y=1)
        lsc_odr = odr.ODR( lsc_data, self.ellipse_model, beta_i)
        lsc_out = lsc_odr.run()
        return lsc_out.beta

class InterpSimple:
    def __init__ ( self,  a,b, qmax,qmin, nphi, raw_img_shape, bin_fac=None ) : 
        '''
        Define a polar image with dimensions: (qmax-qmin) x nphi
        ==========================================================
        a,b           - float, x_center,y_center of cartesian image tht is being interpolated
        qmin,qmax     - cartesianimage will be interpolated from q-qmin, q + qmax
        nphi          - int, azimuthal dimension of polar image (number of azimuthal points along polar image)
        raw_img_shape - int tuple, ydim, xdim of the raw image 
        bin_fac       - float, reduce the size of the cartesian image by this amount

        '''
        self.a = a        # x center
        self.b = b        # y center
        self.qmin = qmin
        self.qmax = qmax  
        self.raw_shape = raw_img_shape   
        self.phis_ring     = arange( nphi ) * 2 * pi / nphi
        self.num_phis_ring = nphi

        self.bin_fac = bin_fac
        
        if bin_fac:
            self.Y,self.X = zoom( np.random.random(raw_img_shape) , # tmp random image
                                    1. / bin_fac, order=1 ).shape 
        else: 
            self.X = raw_img_shape[1] # fast dimension
            self.Y = raw_img_shape[0] # slow dimension

        
        self.Q   = vstack( [ ones(self.num_phis_ring)*iq for iq in arange( self.qmin,self.qmax ) ] )
        self.PHI = vstack( [ self.phis_ring for iq in arange( self.qmin,  self.qmax ) ] )
        self.xring = self.Q*cos(self.PHI-pi) + self.a
        self.yring = self.Q*sin(self.PHI-pi) + self.b

        self.xring_near = self.xring.astype(int) + np.round( self.xring - floor(self.xring) ).astype(int)
        self.yring_near = self.yring.astype(int) + np.round( self.yring - floor(self.yring) ).astype(int)
        self.indices_1d = self.X* self.yring_near + self.xring_near # 'C' style ordering

    def nearest( self, data_img , dtype=np.float32):
        '''return a 2d np.array polar image (fastest method)'''
        if self.bin_fac :
            data_img = zoom( data_img, 1. / self.bin_fac, order=1 ) 
        data = data_img.ravel()
        return data[ self.indices_1d ]


def radial_profile( img, center=None, R = None, norm=None, mask=None, minlength = None) :
    '''
    Return the radial profile of an image
    ======================================
    center    - float tuple, (fast axis center, slow slow center )
    R         - float np.array ,  radial coordinate of each pixel from center 
    norm      - float np.array, number of pixels in each radial bin (needed if mask !- None)
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
            tbin = bincount( R.ravel(), (img*mask).ravel(), minlength = minlength )
            radial_profile = tbin / norm
    else:
        if mask == None:
            tbin = bincount( R.ravel(), img.ravel(), minlength = minlength )
            nr = bincount( R.ravel(), minlength = minlength )
            radial_profile = tbin / nr
        else:
            tbin = bincount( R.ravel(), (img*mask).ravel(), minlength = minlength )
            radial_profile = tbin / norm
    
    return nan_to_num( radial_profile )


class DiffCorr:
    def __init__( self, shots ,delta_shot=None,pre_dif = True ):
        '''
        Initialize a correlation class for doing difference correlations 
        ================================================================
        shots       - 3d float np.array , shape is (num_shot x num_q x num_phi).  
        delta_shot  - set the spacing between "consecutive shots" . A consecutive shot pair will be correlated
                        using the difference correlation method
        '''
        
        if pre_dif:
            self.shotsAB = shots
        else:
            shotsA = shots[0:-delta_shot]
            shotsB = shots[delta_shot:]
            self.shotsAB = shotsA - shotsB
    
    def _remove_highest( self,img, num_high_pix):
        for i in xrange( num_high_pix) :
            x,y = unravel_index( argmax(img) , img.shape  )
            img[x,y] = 0
        return img
    
    def _remove_lowest( self, img, num_high_pix):
        for i in xrange( num_high_pix) :
            x,y = unravel_index( argmin(img) , img.shape  )
            img[x,y] = 0
        return img

    def _fft_autocorr(self, ar):
        n = ar.shape[-1]
        axis = len( ar.shape) -1
        fx = np.fft.rfft( ar, n = n, axis=axis )
        return np.fft.irfft( fx* conjugate( fx), n = n,axis=axis)

    def autocorr(self,num_high=0,num_low=0):
        '''
        Return the difference autocorrelation
        =====================================
        num_high/num_low - int number of high/low pixels to remove before correlating
        '''
        if num_high > 0 or num_low > 0 :
            self.corr = array([self._fft_autocorr(self._remove_lowest(self._remove_highest(s,num_high),num_low) ) for s in self.shotsAB])
        else:
            self.corr = array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return self.corr
    


