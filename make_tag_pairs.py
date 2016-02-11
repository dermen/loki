import sys
import collections

import json
import pandas
import numpy as np

from postproc_helper import is_outlier


class MakeTagPairs:
    def __init__(self, db_pickle, nphi, fixed_qr=False, qrmin=None, 
                    qrmax=None, dqr=1, phi_res=10, rm_mean_outliers=True,
                    rm_mean_thresh=3, min_grp_size=20, signal=None, 
                    min_signal = None):
        """
        'db_pickle' - pandas pickle file from make_db.MakeDatabase
        'nphi'      - number of azimutha points around ring I(phi)
                        should be full range from 0 -2pi
        'fixed_qr' - whether the radial position of each q changing from shot
                    to shot (e.g. if used pk_detect=True in MakeDatabase
        
        If the qr is fluctating, this will group shots according to their
            pk_pos (q  of max intensity along radial profile), and then
            pairings will only be considered for exposures within the same
            group
        
        'qrmin'     - min q to form into groups
        'qrmax'     - max q to form into groups
        'dqr'           - how wide the qr bins are , default is 1
        'min_grp_size' - if grouping according to qr, how big should a group
                        be in order to be analyzed 

        If a shot has pk_pos < qrmin or pk_pos > qrmax, then the shot will 
        not be included in the analysis
        
        'rm_mean_outliers' - whether to remove shot mean outliers
        'rm_mean_thresh'   - threshhold parameter for removing shots
                            make this lower to remove more shots

        'phi_res' - resolution parameter for computing polynomials along 0-2PI 
                    (used for poly comparison of shots)
                default is 10, so the res will be   10 * 2pi/nphi
                make this number higher to speed up comparison computation, 
                but dont make it too high (has'nt been tested)
        'signal'       - tells what kind of signal should be thresholded
                        can be ['mean', 'stdev', 'snr' ]
        'min_signal'   - minimum signal (as determined by the 'signal' parameter) 
                        for a shot to be considered for analysis
        """
        self.db_pickle = db_pickle
        print "\nLoading pandas pickle: %s"%self.db_pickle
        self.df = pandas.read_pickle( self.db_pickle)
        self.fixed_qr = fixed_qr
        self.nphi= nphi
        
        if self.fixed_qr:
            print "Will not form radial groups before pairings..."
        else:
            print "Will form radial groups before pairings..."
            assert (qrmin is not None)
            assert (qrmax is not None)
            assert (dqr is not None)

        self.qrmin = qrmin
        self.qrmax = qrmax
        self.dqr = dqr
        self.phi_res = phi_res
        self.min_grp_size = min_grp_size
        
        assert( signal in [None, 'mean', 'stdev', 'snr'] )
#       whether to apply a secondary signal threshhold
        if signal=='mean':
            "Will use the shot mean filter..."
            self.df['signal'] = self.df.shot_mean 
        elif signal == 'stdev':
            "Will use the shot standard deviation filter..."
            self.df['signal'] = self.df.shot_stdev
        elif signal== 'snr':
            print ("Will use signal to noise filter, be sure you used the",
                    "pk_detect option when making the database or this",
                    "will break")
            self.df['sig'] = self.df.pk_amp / self.df.background_offset

        if signal is not None:
            assert( min_signal is not None)
            n_total = len(self.df)
            self.df = self.df.query('sig >= %f'%min_signal )
            n_removed = n_total - len(self.df)
            print "Removed %d/%d exposures from analysis"%(n_removed,
                                                            n_total)
            
#       exclude exposures with a high/low mean intensity
        if rm_mean_outliers:
            print "Removing shot mean outliers..."
            self._drop_shotmean_outliers(rm_mean_thresh)
        
        self.groups = None
        self.u_gruops = None
        if not fixed_qr: 
            self._group_shots_by_peakpos()

    def _drop_shotmean_outliers(self, thresh):
        outliers = is_outlier( self.df.shot_mean.values, thresh )
        where_outliers = np.where(outliers)[0]
        if where_outliers.size:
            n_total = len(self.df)
            self.df.drop( self.df.index[where_outliers], inplace=True)
            n_removed = n_total-len(self.df)
            print "Removed %d/%d exposures from analysis"%(n_removed,
                                                            n_total)

    def _group_shots_by_peakpos(self):
        """group according to pk_pos parameter, e.g. fitted q111"""
        print "grouping shots according to radial intensity peak position"
        qbins = np.arange( self.qrmin, self.qrmax, self.dqr)
        self.groups = np.digitize( self.df.pk_pos, bins=qbins )
        self.u_groups = np.unique(self.groups) 

    def Make(self, outfile, poly_thresh=3.5):
        """
        make the tag pairings

        outfile - name f the json file which will store the pairings
        poly_thresh - threshhold for detecting outlier chi-square deltas
                        between exposures. If 
                                min_s,s'( chi-square)
                        is still too large, then the pair s,s' wont be 
                        considered. 
        """
        self.outfile = outfile
        self.poly_thresh = poly_thresh

        self.tag_pairs = []
        if not self.fixed_qr:        
            print "Iterating over groups and pairing"
            self._iterate_groups() 

        else:
            print "Pairing all remaining shots in database"
            self.df_g = self.df
            self._pair_group()
        self._save_tag_pairs()

    def _iterate_groups(self):
        """
        Iterate over groups, determined from _group_shots_by_peakpos,
        pairing each group
        """

        for g in self.u_groups:
            self.df_g = self.df.loc[ self.groups==g ]
            if len(self.df_g) < self.min_grp_size:
                continue 
            self._pair_group()    
            
    def _pair_group(self):
        """
        Pairs exposures in a dataframe group according to the
        similarity of their respective polynomials
        """
        ngroup = len(self.df_g)
        
        # polynomial at lower resolution than the ring (it is smooth)
        Px = np.arange(self.nphi)[::self.phi_res]
        Py = np.vstack( [np.polynomial.chebyshev.chebval(Px,c) 
                            for c in self.df_g.cheby_fit_compare])
        
        # polynomial distance measure between shot s and shot s'
        eps = np.array( [ [sum((Py[i]-p)**2)
                             for p in Py ] 
                          for i in xrange(ngroup)]  )
       
        # Add the max to the diagonal, otherwise the diagonal will 
        # register as the minimum 
        epsI = eps.max(1)*np.identity(ngroup)
        eps += epsI
        
        # check if there are any outlier values of eps (if there are
        # they should be positive outliers (greater than the mean)
        outliers = is_outlier(eps.min(1),thresh=self.poly_thresh)

        # pair each shot with the shot that gives minimum eps
        pair_inds = np.array([ (i,eps[i].argmin()) 
                        for i in xrange(eps.shape[0]) 
                        if not outliers[i] ])

        # check whether a tag index appears multiple times
        occurance = collections.Counter(pair_inds[:,1])
        tags = self.df_g.tag.values
        
        # save tag pairs
        self.tag_pairs += [ (tags[i1], tags[i2]) for i1,i2 in pair_inds
                            if occurance[i2] <= 1]

    def _save_tag_pairs(self):
        """
        saves the tag pairings list in a json file
        """
        outfile_ = open( self.outfile, 'w')
        json.dump( self.tag_pairs, outfile_)
        outfile_.close()
        print "Saved tag pairing output in json file: %s"%self.outfile


if __name__ == '__main__':
    db_pickle = '/data/work/mender/interped_178802.pkl'
    qrmin = 20 
    qrmax = 60
    nphi = 5000 # nphi points from 0-2PI in polar image
    makeTagPairs = MakeTagPairs(db_pickle,nphi, qrmin=qrmin, qrmax=qrmax,
                        signal='snr', min_signal=0.2)
   
    outf = '/data/work/mender/loki/interped_178802_pairs.json'
    makeTagPairs.Make(outf)

