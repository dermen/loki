

######
# This is a class  for interpolating correlations
######

import numpy as np

import scipy.interpolate as interp

class InterpCorr:
    def __init__( self, corr, wavlen, q_values, num_psi ):
        '''
        Initialize a interpolation class for interpolating correlations
        ================================================================
        corr       - 2d or 3d float np.array , shape is 
                        (num_shot x num_q x num_phi) or   
                        (num_shot x num_phi) .  it is assumed to contain
                        difference correlation, i.e. each shot is from a pair
                        of exposures.
                        
        wavlen     - 1d float np.array, shape is (num_shot)
                    angstrom
        
        q_values   - 2d flost np.array, shape is (num_q x 2)
                     inv angstrom
        
        num_psi    - int, number of cos(psi) values after interpolation. 
                    cos(psi) = [-1,1]
        '''
        self.corr = corr
        self.wavlen = wavlen
        self.q_values = q_values
        self.num_psi = num_psi
    
    def interpolate(self):
        """
        Returns interpolated correlation data, and cos(psi) for all data
        ================================================================
        If cos(psi) is out of bounds for the data before interpolation, 
        interpolated data = numpy.inf at those cos(psi) values
        
        Returns:
        
        interp_corr - 2d or 3d float np.array , shape is 
                        (num_shot x num_q x num_psi) or   
                        (num_shot x num_psi) .
        new_cpsi    - 1d array, shape = (num_psi)
                      cos(psi) values conrresponding to the interpolated
                      correlation data. 
        """
        # get new cpsi for after interpolation
        new_cpsi = np.linspace( -1, 1, num=self.num_psi, endpoint=True)
        
        if len(self.corr.shape)==3:
            interp_corr = np.zeros( (self.corr.shape[0], self.corr.shape[1], self.num_psi) )
        else:
            interp_corr = np.zeros( (self.corr.shape[0], self.num_psi) )        
        
        for qs in self.q_values:
            for idx, w in enumerate(self.wavlen):
                num_phi = self.corr[idx].shape[-1]
                cpsi = InterpCorr.get_cpsi( num_phi, w, qs[0], qs[1] )
                
                interpolator = interp.interp1d( cpsi, self.corr[idx],
                bounds_error= False, fill_value=np.inf)
                interp_corr[idx] = interpolator(new_cpsi)
        
        return interp_corr, new_cpsi
            
    @staticmethod
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
        
        
    