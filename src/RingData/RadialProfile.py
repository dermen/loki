import numpy as np

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

