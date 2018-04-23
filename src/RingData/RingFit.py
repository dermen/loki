import numpy as np
from scipy import odr

from loki.RingData import RadialProfile

class RingFit:
    def __init__( self, img ):
        '''
        Initialize a class for fitting shapes to images with shapes on them
        ====================================================================
        img - 2d float np.array
        '''

        self._set_circle_model()
        self._set_circle_model_fast()
        self._set_ellipse_model()
        
        self.img = np.copy(img)
        self._store_1D_image_index_arrays()


    def _set_circle_model_fast( self):
        f_circle  = lambda beta,x : ( x[0] - beta[0] )**2 \
                                  + ( x[1] - beta[1])**2 - beta[2]**2
        self.circle_model_fast = odr.Model(f_circle, 
            implicit=True,
            estimate=self.calc_estimate,
            fjacd=self.circle_jacobian_data, 
            fjacb=self.circle_jacobian_beta)

    def _set_circle_model(self):
#       Equation for a circle        
        f_circle  = lambda beta,x : ( x[0] - beta[0] )**2 \
                                  + ( x[1] - beta[1])**2 - beta[2]**2
        self.circle_model  = odr.Model( f_circle, implicit=True)

    def _set_ellipse_model(self):
#       Equation for an ellipse
        f_ellipse = lambda beta,x : (1/beta[2]/beta[2])*( x[0] -beta[0])**2 \
                                  + (1/beta[3]/beta[3])*( x[1] -beta[1])**2-1
        self.ellipse_model = odr.Model( f_ellipse, implicit=True)

    def _store_1D_image_index_arrays(self):
#       make 2d and 1d index arrays       
        self.y, self.x = np.indices( self.img.shape )
        self.x1D   = self.x.ravel()
        self.y1D   = self.y.ravel()
        

    def fit_circle_fast( self, beta_i, num_fitting_pts=5000, 
                ring_width=20 , num_high_pix=20, 
                return_mean = False):
        '''
        Fit a circle to a ring image
        ============================== 
        beta_i          - float tuple , ( x_center, y_center,radius )
        num_fitting_pts - int, number of pixels to include in fit 
        ring_width      - int, width ring-like region on image that 
                            contains ring of interest (pixels units)
        num_high_pix    - int, remove this many pixels before 
                            running fit (removes artificial high pixels 
                            that might bias the fit)
        '''

        self.ring_width = ring_width
        self._prepare_fitting_framework( beta_i, num_high_pix, num_fitting_pts)
        self._fit_a_model_fast(self.circle_model_fast, param_guess=beta_i)
    
        x,y = self._get_xy()
        a,b,r  = self.beta_fit
        Ri = np.sqrt( (x-a)**2 + (y-b)**2)
        self.residual = np.sum((Ri-r)**2)
        mean_intens = self.img1D[ self.fit_indices].mean()
        return self.beta_fit, self.residual, mean_intens

    def fit_circle( self, beta_i, num_fitting_pts=5000, 
                ring_width=20 , num_high_pix=20, 
                return_mean = False):
        '''
        Fit a circle to a ring image
        ============================== 
        beta_i          - float tuple , ( x_center, y_center,radius )
        num_fitting_pts - int, number of pixels to include in fit 
        ring_width      - int, width ring-like region on image that 
                            contains ring of interest (pixels units)
        num_high_pix    - int, remove this many pixels before 
                            running fit (removes artificial high pixels 
                            that might bias the fit)
        '''

        self.ring_width = ring_width
        self._prepare_fitting_framework( beta_i, num_high_pix, num_fitting_pts)
        self._fit_a_model(self.circle_model, param_guess=beta_i)

        x,y = self._get_xy()
        a,b,r  = self.beta_fit
        Ri = np.sqrt( (x-a)**2 + (y-b)**2)
        self.residual = np.sum((Ri-r)**2)
        
        mean_intens = self.img1D[ self.fit_indices].mean()

        return self.beta_fit, self.residual, mean_intens

    def _get_xy(self):
        """returns the x and y points used in _fit_a_model"""
        return self.pts[0], self.pts[1]

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
        
        self.ring_width = ring_width
        self._prepare_fitting_framework( beta_i, num_high_pix, num_fitting_pts)
        self._fit_a_model( self.ellipse_model, param_guess=beta_i)
        return self.beta_fit


    def _prepare_fitting_framework(self, ring_guess, num_high_pix, 
                                    num_fitting_pts):
#       set the 1D image
        self.img1D = np.copy(self.img.ravel())
        self._mask_1D_image(ring_guess)
        self._remove_highest(num_high_pix)
        num_fitting_pts = self._check_num_fitting_points(num_fitting_pts)
#       find indices of fit pixels in order of decreasing intensity 
        self.fit_indices = np.argsort(self.img1D)[::-1][: num_fitting_pts]

    def _mask_1D_image(self, beta_i):
#       radius of each pixel 
        radius_value = np.sqrt((self.x1D - beta_i[0])**2 + \
                            (self.y1D - beta_i[1])**2 )
        self.img1D[ radius_value > beta_i[2] + self.ring_width/2 ] = 0
        self.img1D[ radius_value < beta_i[2] - self.ring_width/2 ] = 0
        

    def _remove_highest( self, num_high_pix):
#       removes the highest pixels so they don't corrupt the fit 
#           (e.g. bad pixels)
        high_inds = np.argsort( self.img1D )[:num_high_pix]
        self.img1D[ high_inds ] = 0

    def _check_num_fitting_points(self, num_fitting_pts):
        num_pts = np.where( self.img1D > 0 )[0].shape[0]
        if num_fitting_pts >= num_pts :
            num_fitting_pts = num_pts / 2
        return num_fitting_pts

       
    @staticmethod
    def circle_jacobian_beta( beta, x):
        xc,yc,r = beta
        xi,yi = x
        df_db    = np.empty(( len(beta), x.shape[1]))
        df_db[0] =  2*(xc-xi)                     # d_f/dxc
        df_db[1] =  2*(yc-yi)                     # d_f/dyc
        df_db[2] = -2*r                           # d_f/dr
        return df_db

    @staticmethod
    def circle_jacobian_data(beta, x):
        xc,yc,r = beta
        xi,yi = x
        df_dx    = np.empty_like( x )
        df_dx[0] =  2*(xi-xc)                     # d_f/dxi
        df_dx[1] =  2*(yi-yc)                     # d_f/dyi
        return df_dx
   
    @staticmethod
    def calc_estimate(data):
        xc0, yc0 = data.x.mean(axis=1)
        r0 = np.sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
        return xc0, yc0, r0


    def _fit_a_model_fast(self, model, param_guess):
#       use odr module to fit data to model
        self.pts = np.row_stack( [self.x1D[ self.fit_indices], 
                            self.y1D[ self.fit_indices]])
        
        lsc_data = odr.Data( self.pts , y=1)
        lsc_odr   = odr.ODR(lsc_data, model)#  param_guess)
        lsc_odr.set_job(deriv=3) 
        lsc_odr.set_iprint(iter=1, iter_step=1)
        lsc_out = lsc_odr.run()
        self.beta_fit = lsc_out.beta

    def _fit_a_model(self, model, param_guess):
#       use odr module to fit data to model
        self.pts = np.row_stack( [self.x1D[ self.fit_indices], 
                            self.y1D[ self.fit_indices]])
        lsc_data = odr.Data( self.pts , y=1)
        lsc_odr = odr.ODR( lsc_data, model, param_guess)
        lsc_out = lsc_odr.run()
        self.beta_fit = lsc_out.beta


    def fit_circle_slow(self, beta_i, ring_scan_width, center_scan_width, 
                            resolution=1 ):
        """
        Description
        ===========
        Here we want to find where the forward x-ray beam intersects 
        the detector, (x_center, y_center). 
        
        The radial profile of the diffraction pattern should exhibit
        scattering peaks, and the magnitude of the radial profile will depend
        on the chosen point for the center. Therefore, by calculating many radial profiles
        for a range of centers, and choosing the radial profile with the maximum magnitude
        across a specific radial range( e.g. across a Bragg ring), then we can
        optimize the center.

        Parameters
        ==========

        `beta_i`    tuple of (x_center, y_center, ring_position)
                    all in pixel units
        `ring_scan_width`   how many pixels to scan for ring maxima
        `center_scan_width`  how many pixels to scan in horizontal and vertical 
                            diredtions for ring maxima
        
        Returns
        =======
        tuple of optimized center parameters (x_center, y_center, max_radius)
        """

        self.resolution = resolution

        self.ring_radius_guess = round(beta_i[2])
        self.center_x_guess, self.center_y_guess =  round(beta_i[0]), \
                                        round( beta_i[1] )  
        
        self.RadPro = RadialProfile( center=(self.center_x_guess, \
                                        self.center_y_guess), 
                                img_shape=self.img.shape, mask=None, 
                                minlength=self.img.shape[0] )

        self._set_radial_scan_range(ring_scan_width)
        self._define_possible_center_coordinates(center_scan_width)
        self._store_profiles_for_each_center()
        self._store_maxima_of_each_profile()
        self._find_center_with_maximum_ring_profile()
        self._set_max_ring_profile()
       
        return self._get_fit_parameters()
        

    def _set_radial_scan_range(self, ring_scan_width):
        self.ring_scan_start = self.ring_radius_guess - int( ring_scan_width / 2 )
        self.ring_scan_stop = self.ring_radius_guess + int( ring_scan_width / 2 )

        
    def _define_possible_center_coordinates(self, center_scan_width):
        num_center_scan_points = int( center_scan_width / self.resolution) 
        scan_range_x = np.linspace(self.center_x_guess- \
                                        center_scan_width / 2. , 
                                    self.center_x_guess + \
                                        center_scan_width / 2. , 
                                    num_center_scan_points )
        scan_range_y = np.linspace(self.center_y_guess- \
                                        center_scan_width / 2. , 
                                    self.center_y_guess + \
                                        center_scan_width / 2. , 
                                    num_center_scan_points )
        
        self.possible_centers = [ (x,y) for x in scan_range_x
                            for y in scan_range_y]

    def _store_profiles_for_each_center(self):
        self.possible_ring_profiles = []
        for center in self.possible_centers:
            self.RadPro.update_center( center)
            radial_profile = self.RadPro.calculate(self.img) 
            ring_profile = radial_profile[ int(self.ring_scan_start):\
                                            int(self.ring_scan_stop)]
            self.possible_ring_profiles.append( ring_profile)
    
    def _store_maxima_of_each_profile(self):
        self.ring_profile_maxima = [ max( ring_profile) 
                        for ring_profile in self.possible_ring_profiles ]
        
    def _find_center_with_maximum_ring_profile(self):
        self.fit_center = self.possible_centers[ np.argmax(self.ring_profile_maxima) ]
        
    def _set_max_ring_profile(self):
        self.max_profile = self.possible_ring_profiles[ \
                                np.argmax(self.ring_profile_maxima) ]
    
    def _get_fit_parameters(self):
        fit_radius = np.argmax( self.max_profile ) + self.ring_scan_start
        return ( self.fit_center[0], self.fit_center[1], fit_radius )


