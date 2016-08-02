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

    def autocorr(self): #,num_high=0,num_low=0):
        '''
        Return the difference autocorrelation
        =====================================
        num_high/num_low - int number of high/low pixels to remove 
        before correlating
        '''
        self.corr = np.array( [ self._fft_autocorr(s) for s in self.shotsAB ] )
        return self.corr
        
    def weighted_average(self, autocorr, q_value, wavelen, interpolate = True, num_phi = 2500, 
    num_iter=150, learning_rate = None, check_converg=True):
        '''
        Return weights for averaging autocorr that maximizes symmetry around pi in the weighted average
        =====================================
        autocorr      - np.array, shape = (num of shots, num of phi)
        q_value       - float, magnitude of q in inverse angstrom
        wavelen       - float, in angstrom
        interpolate   - default true, will interpolate data to have evenly spaced cos psi. Otherwise, positive
                        cos psi are mapped to the negative cos psi with the closest absolute value when 
                        computing symmetry of a correlation
        num_phi       - int, if interpolate, is the number of phi/cos psi the interpolated data have
        num_iter      - int, maximum number of iterations to run for the gradient descent
        learning_rate - float, rate of the gradient descent, default None estimate using the sum of squares of magnitudes of the autcorr
        check_coverg - default true, if true, returns the asymmetry of the weighted average and the change 
                        in weights for every iteration
        
        return:
        =====================================
        weights       - normalized weights for each shot in autocorr, numpy.dot(weights, autocorr) gives the weighted average
        cpsi          - cos psi corresponding to the weighted averaged 
        autocorr      - interpolated autocorr, only returned if interpolate 
        asyms         - asymmetry of the weighted average for every iteration, only returned if check_converg
        change_in_weights - he change in weights for every iteration, only returned if check_converg
        
        '''
        if not interpolate:
            num_phi = autocorr.shape[1]
            
        if learning_rate==None:
            # estimate learning rate
            learning_rate = 1.0/ autocorr.shape[0] /np.sum(np.mean(autocorr,axis=0)**2)
        
        # calculate cos psi
        cpsi = self.get_cpsi(autocorr.shape[1], q_value, wavelen)
        
        if interpolate:
            autocorr, cpsi = self._interpolate_autocorr(autocorr,num_phi,cpsi)
            
            # map indices that are suppose to be equal by symmetry
            x=range(0,num_phi/2-1,1)
            y=range(num_phi-2,num_phi/2-1,-1)
            
            # mapping indices of cpsi for left to right-hand side of autocorr
            mapping = np.array(zip(x,y))
#             try:
#                 assert ( np.round(cpsi[ mapping[i,0] ], 4) == -np.round(cpsi[ mapping[i,1] ], 4) for i in range(mapping.shape[0]) )
#             except AssertionError: 
#                 print cpsi [ mapping [0,0]], cpsi [ mapping [0,1]]

        else:
            # if not interpolating, match cpsi closest in absolute value
            half=cpsi[:num_phi/2]
            # find where cpsi ~ 0 is
            split=np.where(half>0)[0][-1]
            # length is number of negative cpsi values that we can try to find a match
            length=len(half)-split-1
            # ind of cpsi going from min(cpsi) to 0 
            ind=np.arange(len(half)-1,split,-1)

            # mapping
            # find the match of a posivite cpsi values that is closest in absolute value to every negative cpsi value
            match = np.array( [ int( (ind - length)[np.argmin( (half[ind-length] + half[i])**2.)] ) for i in ind] )
            
            assert ( len(ind) == len(match) )
            
            # mapping indices of cpsi for left to right-hand side of autocorr
            mapping = np.array(zip(ind,match))
        
        print ("begining gradient descent: learning_rate = %g"%learning_rate)
        weights, asyms = self._iter_gradient_descent(autocorr, mapping, num_iter,learning_rate)
            
        if interpolate:
            if check_converg:
                return weights, cpsi, autocorr, asyms # , change_in_weights
            else:
                return weights, cpsi, autocorr
        else:
            if check_converg:
                return weights, cpsi, asyms #, change_in_weights
            else:
                return weights, cpsi
    
    def _iter_gradient_descent(self, autocorr, mapping, num_iter, learning_rate):
        
        # initiate weights by assigning uniform weights
        weights=np.ones(autocorr.shape[0])/autocorr.shape[0]
        
        asyms = np.zeros(num_iter)
#         change_in_weights=np.zeros(num_iter)
        
        # compute a threshold for convergence: 0.1% of total variance in the average
        thresh = np.var( np.dot(weights,autocorr) ) * autocorr.shape[1] * 0.1/100.

        gradient = np.zeros(weights.shape)
         # iterate through gradient descent
        for k in range(num_iter):
            old_weights = weights.copy()
            ave_autocorr = np.dot(weights,autocorr) 
            asymmetry = np.sum((ave_autocorr[mapping[:,0]]-ave_autocorr[mapping[:,1]])**2)
            asyms[k]=asymmetry
            
            if asymmetry <= thresh:
                print "Convergence in weighted averaging achieved after %d iterations"%k
                asyms = asyms[:k+1]
#                 change_in_weights=change_in_weights[:k+1]
                break


            for j in range(gradient.shape[0]):
                gradient[j] = np.sum((autocorr[j][mapping[:,0]]-autocorr[j][mapping[:,1]])*\
                             2.0*(ave_autocorr[mapping[:,0]]-ave_autocorr[mapping[:,1]]))*learning_rate
            weights = weights-gradient
            weights[weights<0]=0.0
            # normalize weights
            weights=weights/np.sum(weights)

#             change_in_weights[k]=np.sum((weights-old_weights)**2.0)
        
        if asyms[-1] > thresh:
            print "WARNING: weighted averaging did NOT converage after %d iterations."%num_iter
        
        return weights, asyms #, change_in_weights
    
    def _interpolate_autocorr(self, autocorr, num_phi, cpsi):
        # interpolation
        step = -np.min(cpsi) * 2./ num_phi
        new_cpsi = np.arange(np.min(cpsi)+step,-np.min(cpsi),step)
        inter_dif_cor = np.zeros((autocorr.shape[0],new_cpsi.shape[0]))
        
        print ('interpolating autocorrelations...')
        for i, a in enumerate(autocorr):
            interpolated_autocorr = interp.interp1d(cpsi,a)
            inter_dif_cor[i] = interpolated_autocorr(new_cpsi)
        
        return inter_dif_cor, new_cpsi
    
    def get_cpsi(self, num_phi, q_value, wavelen):
    
        phis = np.arange( num_phi ) * 2 * np.pi / num_phi
        th = np.arcsin( q_value * wavelen / 4. / np.pi )
        cpsi = np.cos(phis) * np.cos(th)**2. + np.sin(th)**2. 
        
        return cpsi