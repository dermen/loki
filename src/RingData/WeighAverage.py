
import numpy as np
import sys

from loki.RingData import InterpCorr



class WeighAverage:
    def __init__( self, corr, cpsi, interpolated = True):
        """
        Initialize a weighted average class
        ================================================================
        corr         - 2D np.array (num_shots x num_psi)
                    all correlations needed to be average
                    
        cpsi         - 1D np.array (num_psi)
                     cos psi values, assumed to be interpolated, range from -1 to 1
        
        interpolated - Bool, default True, flags that the corr is interpolated
        """
        
        # make a 1D mask that filters out all inf to get 2D array for corr
        # inf is a result of out-of-range values during interpolation
        # see InterpCorr for details
        self._make_mask(corr)
        self.corr = corr[:,self.mask]
        assert ( ~np.isinf( np.mean(self.corr) ) )
        self.cpsi = cpsi[self.mask]
        self.weights = np.ones(self.corr.shape[0])/self.corr.shape[0]
        
        self.interpolated = interpolated
        
        # guess initial parameters for 
        self.num_iter = 30
        # compute a threshold for convergence: 1% of total variance in the average
        self.thresh = np.var( np.dot(self.weights,self.corr) ) * self.corr.shape[1] * 1/100.
        self.learning_rate = 1.0/ self.corr.shape[0] /np.sum(np.mean(self.corr, axis=0)**2.0)
        
        # initialize mapping of indices
        self._map_symmetry_indices()
    
    
    def set_params(self, num_iter, learning_rate, thresh):
        """
        Change the gradient descent fit paramters
        ================================================================
        num_iter       - int, max number of interations
        
        learning_rate  - float, scaling factor for gradient descent steps
        
        thresh         - float, asymmetry convergence threshold 
        """
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.thresh = thresh
  
    def fit(self):
        """
        Do gradient descent to find weights for maximally symmetric average
        ================================================================
        
        Returns:
        weights - 1D np.array, (num_shots), weight for each shot in corr
        
        asyms   - 1D np.array, (num_iter), total amount of asymmetry at each iteration
                  Should converge to a minimum for the appropriate fit params. 
                  Use set_params to adjust learning_rate, num_iter, and thresh if needed.
        """
        print ("begining gradient descent: ")
        print ("num_iter =%d, learning_rate = %g, thresh = %g"%(self.num_iter, self.learning_rate, self.thresh))
        
        self.weights = np.ones(self.corr.shape[0])/self.corr.shape[0]
        asyms = np.zeros(self.num_iter)
        gradient = np.zeros(self.weights.shape)
        
         # iterate through gradient descent
        for k in range(self.num_iter):
            old_weights = self.weights.copy()
            ave_corr = np.dot( self.weights, self.corr ) 
            asymmetry = np.sum(( ave_corr[self.mapping[:,0]]- ave_corr[self.mapping[:,1]] )**2.0)
            asyms[k]=asymmetry
            
            if asymmetry <= self.thresh:
                print ("Convergence in weighted averaging achieved after %d iterations"%k)
                asyms = asyms[:k+1]
                break


            for j in range(gradient.shape[0]):
                gradient[j] = np.sum( (self.corr[j][self.mapping[:,0]]-self.corr[j][self.mapping[:,1]]) *\
                             2.0 * (ave_corr[self.mapping[:,0]]-ave_corr[self.mapping[:,1]]) ) * self.learning_rate
            self.weights = self.weights-gradient
            self.weights[self.weights<0]=0.0
            # normalize weights
            self.weights=self.weights/np.sum(self.weights)
        
        if asyms[-1] > self.thresh:
            print( "WARNING: weighted averaging did NOT converage after %d iterations."%self.num_iter)
        
        return self.weights, asyms 


    def _make_mask(self, corr):
        inf_truth_table = np.isinf(corr)
        self.mask = ~np.mean(inf_truth_table,axis=0,dtype=bool)

    def _map_symmetry_indices(self):
        num_psi = len(self.cpsi)
        
        if self.interpolated:
            try:
                end_point = np.where(np.round(self.cpsi[0]+self.cpsi, 5) == 0 )[0][0]
            except IndexError:
                print ("ERORR!!! Cannot find match for minimum cos psi value %g"%np.min(self.cpsi))
                sys.exit()
                
            # find where cpsi = zero or close to zero is
            mid_point = np.argmin(np.abs(self.cpsi))
            x = np.arange(0, mid_point+1, 1)
            y = np.arange( end_point,mid_point, -1)

            self.mapping = np.array( list( zip(x,y) ))
            
            # check that mapping has been done correctly
            assert ( ( np.round(self.cpsi[self.mapping[:,0]]+ self.cpsi[self.mapping[:,1]],5) == 0.0 ).all() )
        
        else:        
            #############################################################################
            ##### Deprecated code: all corr should be interpolated in the future#########
            # if data passed is not interpolated, match cpsi closest in absolute value
            half=self.cpsi[:num_psi/2]
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
            self.mapping = np.array( list( zip(ind,match)))
            
            #############################################################################
    

    @staticmethod    
    def degree_of_ass(diffcorr, wavelen, q_values):
        """
        Parameters
        ==========

        `diffcorr`      has shape Nq x Nphi
        `wavelen`   is a float in angstroms
        `q_values`  is a list of q_values corresponding to 
                    the Nq dimension of diffcorr
        """
        
        Nq, Nphi = diffcorr.shape

#       emphasis on "ass" in "ass"-ert
        assert( Nq == q_values.shape[0] )

        interp_cor = InterpCorr( diffcorr[None,:,:], 
            wavlen=[wavelen],
            q_values= np.vstack((q_values, q_values) ).T,
            num_psi=Nphi )
       
        cors, cos_psi = interp_cor.interpolate()

        all_ass = []
        for iq in xrange(Nq):
            wa = WeighAverage(cors[:,iq], cos_psi)
            all_ass.append ( np.sum((wa.corr[:,wa.mapping[:,0]] \
                                     - wa.corr[:,wa.mapping[:,1]])**2.0, axis = 1))
        
        return np.mean( all_ass)

