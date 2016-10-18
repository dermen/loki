import numpy as np

import scipy.interpolate as interp

class DiffCorr:
    def __init__( self, shots ,delta_shot=None,pre_dif=True, generate_mode=False ):
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
            assert( not generate_mode)
            if delta_shot is None:
                delta_shot = 1
            shotsA = shots[0:-delta_shot]
            shotsB = shots[delta_shot:]
            self.shotsAB = shotsA - shotsB
        self.generate_mode = generate_mode
    
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
