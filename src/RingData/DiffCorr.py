import numpy as np

import scipy.interpolate as interp

class DiffCorr:
    def __init__( self, shots, q_values, k
        ,delta_shot=None,pre_dif=True
        ,generate_mode=False ):
        '''
        Initialize a correlation class for doing difference correlations 
        ================================================================
        shots       - 3d float np.array , shape is 
                        (num_shot x num_q x num_phi).
                        phi values are linspaced between 0 and 2pi
        q_values    - 1d float np.array, q values for the polar intensities
        k           - float, wave number of the beam
        delta_shot  - set the spacing between "consecutive shots" . 
                        A consecutive shot pair will be correlated
                        using the difference correlation method
        pre_dif     - default True, assume shots is difference in 
                        intensities
        generate_mode - default False, if True, correlations are 
                        returned as generators. Use the corresponding
                        generator functions to compute auto or 
                        crosscorrs.

        '''
        
        if pre_dif:
            self.shotsAB = shots
        else:
            assert( not generate_mode)
            if delta_shot is None:
                delta_shot = 1
            shotsA = shots[0:-delta_shot]
            shotsB = shots[delta_shot:]
            self.shotsAB = shotsA - shotsB
        self.generate_mode = generate_mode

        self.num_phi = self.shotsAB.shape[-1]
        self.phi_values = np.linspace(0., 2.*np.pi, self.num_phi)
        
        self.num_q = self.shotsAB.shape[-2]
        assert (self.num_q == q_values.size)
        self.q_values = q_values
        self.k = k


    def correct_polarization(self, yaxis_polarization):
        """
        Applies a polarization correction to the rings.
        
        Parameters
        ----------
        yaxis_polarization : float
            The fraction of the beam polarization in the vertical/y plane.
            For synchrotron sources, this is the ''out-of-plane'' polarization.
            
        Citations
        ---------
        ..[1] Hura et. al. J. Chem. Phys. 113, 9140 (2000); doi10.1063/1.1319614
        ..[2] Jackson. Classical Electrostatics.
        """
        
        if self.k == 0.0:
            raise ValueError('Rings.k is 0.0, which indicates simulated data --'
                             ' no polarization correction possible.')

        correctn = np.zeros((self.num_q, self.num_phi))

        for i,q in enumerate(self.q_values):
            theta     = np.arcsin( q / (2.0 * self.k) )
            sin_theta = np.sin(2.0 * theta)
            correctn[i,:]  = (1.-yaxis_polarization) * \
                             ( 1. - np.square(sin_theta * np.cos(self.phi_values)) ) + \
                             yaxis_polarization  * \
                             ( 1. - np.square(sin_theta * np.sin(self.phi_values)) )
            
        for pi in self.shotsAB:
            pi[:,:] /= correctn[:,:]

        self.polarization_correted = True

        return
    
    def correct_solid_angle(self):
        """
        use the geometry of the polar pixels to correct for the fact that some of them
        cover larger solid angles than others
        """

        return
    
    def _fft_autocorr(self, ar):
        n = ar.shape[-1]
        axis = len( ar.shape) -1
        fx = np.fft.rfft( ar, n = n, axis=axis )
        return np.fft.irfft( fx* np.conjugate( fx), n = n,axis=axis)
    
    def _fft_crosscorr(self, ar, ar_2 ):
        n = ar.shape[-1]
        axis = len( ar.shape) -1
        fx = np.fft.rfft( ar, n = n, axis=axis )
        fx_2 = np.fft.rfft( ar_2, n = n, axis=axis )
        return np.fft.irfft( fx* np.conjugate( fx_2), n=n, axis=axis)

    def autocorr(self): 
        '''
        Return the difference autocorrelation
        =====================================
        '''
        assert( not self.generate_mode)
        corr = np.array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return corr
    
    def autocorr_generator(self): 
        '''
        Return the difference autocorrelation
        =====================================
        '''
        assert( self.generate_mode)
        for s in self.shotsAB:
            yield self._fft_autocorr(s)  

    def crosscorr(self, qindex): 
        '''
        Correlates ring denoted by qindex with 
        every other ring (including itself)
        '''
        assert( len(self.shotsAB.shape) == 3)
        assert( not self.generate_mode)

        crosscorrs = []
        for shot in self.shotsAB:
#               `shot` has shape (Nq x Nphi)
            qring = np.vstack( [shot[qindex]]*shot.shape[0] ) 
            shot_crosscorr = self._fft_crosscorr( qring, shot )
            crosscorrs.append( shot_crosscorr)
        return np.array(crosscorrs)
   
    def crosscorr_generator(self, qindex): 
        '''
        Correlates ring denoted by qindex with 
        every other ring (including itself)
        '''
        assert( len(self.shotsAB.shape) == 3)
        assert( self.generate_mode)

        for shot in self.shotsAB:
            qring = np.vstack( [shot[qindex]]*shot.shape[0] ) 
            shot_crosscorr = self._fft_crosscorr( qring, shot )
            yield shot_crosscorr