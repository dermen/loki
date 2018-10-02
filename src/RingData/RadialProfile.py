import numpy as np
from scipy.stats import binned_statistic

from loki.RingData import RingFetch

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
        self._set_R_sorted_idx()

        self.wavelen = None
        self.pixsize = None
        self.detdist = None
        self.factor = None

    def _set_R_sorted_idx(self):
        self.R1 = self.R.ravel()
        self.sort_idx = np.argsort( self.R1 )
        Rsort = self.R1[self.sort_idx]
        self.Rsort_bin_edge = np.where( Rsort[1:] - Rsort[:-1] == 1)[0] + 1
        self.Rsort_bin_edge =  np.array( \
            [0] + list( self.Rsort_bin_edge) + [ len( self.R1) ] )
    
    def _set_R(self):
        self.R = np.sqrt((self.Y-self.y_center)**2 + \
                        (self.X-self.x_center)**2)
        self.R = self.R.astype(int)
        self.Rbins = np.arange( self.R.max()+1) 
    
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
    
    def calculate2(self, img, how="mean", bins=None):
        """
        img:   2-dimensional image to be radially binned
        how:   what metric to use
        bins:  which bins to use, only applies to how=='max' case
        returns the 1-dimensional radial profile of img
        """
        if how=="mean":
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
        elif how=="max":
            if bins is not None:
                result =  binned_statistic( self.R.ravel(), values=img.ravel(), 
                                                bins=bins, statistic='max' )
            else:
                result =  binned_statistic( self.R.ravel(), values=img.ravel(), 
                                                bins=self.Rbins, statistic='max' )
            radial_profile = result.statistic
        
        elif how=='fast_max':
            I1d = img.ravel()
            #Nbin = len( self.sort_idx)
            #radial_profile = np.zeros( Nbin)
            radial_profile = [] 
            for i_b,(b1,b2) in enumerate(zip( self.Rsort_bin_edge[:-1], self.Rsort_bin_edge[1:])):
                radial_profile.append( max( I1d[ self.sort_idx[b1:b2] ] )) 

        else:
            print("how must be mean, max or fast_max")
            return None

        return np.nan_to_num(radial_profile)

    def calculate_using_fetch( self, img, radii, wavelen=None, 
            detdist=None, pixsize=None, factor=None, 
            q_resolution=.01, phi_resolution=10,index_query_fname=None):
        """
        Calculates a radial profile using a much slower method that 
        adjusts the output for solid angle
        """

        if index_query_fname is None:
            interp_method='floor'
        else:
            interp_method='nearest'

        if wavelen is None:
            assert( self.wavelen is not None)
            w = self.wavelen
        else:
            w = wavelen
        if detdist is None:
            assert( self.detdist is not None)
            d = self.detdist
        else:
            d = detdist
        if pixsize is None:
            assert( self.pixsize is not None)
            p = self.pixsize
        else:
            p = pixsize
        if factor is None:
            assert( self.factor is not None)
            f = self.factor
        else:
            f = factor

        fetch = RingFetch( 
            self.x_center, 
            self.y_center, 
            img=img,
            mask=self.mask,
            wavelen=w,
            detdist=d,
            pixsize=p,
            photon_conversion_factor=f,
            q_resolution=q_resolution,
            phi_resolution=phi_resolution,
            interp_method=interp_method,
            index_query_fname=index_query_fname)

        rings = np.ma.masked_array( data=np.zeros( (len(radii), fetch.num_phi_nodes) ) ) 
        for i,r in enumerate(radii):
            #print("    Fetching ring pixels at radius r=%d"%r)
            rings[i] = fetch.fetch_a_ring(radius=r)
        rings.mask = np.isnan(rings.data)
        return rings.mean(1)

    def update_center( self, new_center):
        self.x_center = new_center[0]
        self.y_center = new_center[1]
        self._set_R()
        self._set_normalization()

    def set_params( self, wavelen, detdist, pixsize, factor):
        self.wavelen = wavelen
        self.detdist = detdist
        self.pixsize = pixsize
        self.factor = factor

