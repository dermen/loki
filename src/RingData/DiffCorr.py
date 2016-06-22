import numpy as np

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
        self.corr = np.array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return self.corr

