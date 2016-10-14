import numpy as np

import scipy.interpolate as interp

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
        self.corr = np.array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return self.corr
    
    def crosscorr(self, qindex): 
        '''
        Correlates ring denoted by qindex with 
        every other ring (including itself)
        '''
        assert( len(self.shotsAB.shape) == 3)

        crosscorrs = []
        for shot in self.shotsAB:
#           `shot` has shape (Nq x Nphi)
            qring = np.vstack( [shot[qindex]]*shot.shape[0] ) 
            shot_crosscorr = self._fft_crosscorr( qring, shot )
            crosscorrs.append( shot_crosscorr)
        return np.array(crosscorrs)

