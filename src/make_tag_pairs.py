import sys
import collections

import json
import pandas
import numpy as np

from loki.utils import postproc_helper as helper

class MakeTagPairs:
    def __init__(self, db_pickle, fixed_qr=False, qrmin=None, 
                    qrmax=None, dqr=1, poly_stride=10, rm_mean_outliers=True,
                    rm_mean_thresh=3, min_grp_size=20, signal=None, 
                    min_signal = None, max_signal = None):
        """
        'db_pickle' - pandas pickle file from make_db.MakeDatabase
        'fixed_qr' - whether the radial position of each q changing from
                     shot to shot (e.g. if used pk_detect=True 
                     in MakeDatabase
        
        If the qr is fluctating, this will group shots according to their
            pk_pos (q  of max intensity along radial profile), and then
            pairings will only be considered for exposures within the same
            group
        
        'qrmin'     - min q to form into groups
        'qrmax'     - max q to form into groups
        'dqr'           - how wide the qr bins are , default is 1
        'min_grp_size' - if grouping according to qr, how big should a 
                        group be in order to be analyzed 

        If a shot has pk_pos < qrmin or pk_pos > qrmax, then the shot will 
        not be included in the analysis
        
        'rm_mean_outliers' - whether to remove shot mean outliers
        'rm_mean_thresh'   - threshhold parameter for removing shots
                            make this lower to remove more shots

        'poly_stride' - resolution parameter for computing polynomials along
                     0-2PI 
                    (used for poly comparison of shots)
                default is 10, so the res will be   10 * 2pi/nphi
                make this number higher to speed up comparison computation, 
                but dont make it too high (has'nt been tested)
        'signal'       - tells what kind of signal should be thresholded
                        can be ['mean', 'stdev', 'snr' ]
        'min_signal'   - minimum signal (as determined by the 'signal' 
                        parameter) 
        'max_signal'   - maximum signal (as determined by the 'signal'
                         parameter) 
                        for a shot to be considered for analysis
        """
        self.db_pickle = db_pickle
        print ("\nLoading pandas pickle: %s"%self.db_pickle)
        self.df = pandas.read_pickle( self.db_pickle)
        self.fixed_qr = fixed_qr
        self.nphi = self.df.num_phi[0]
        
        if self.fixed_qr:
            print ("Will not form radial groups before pairings...")
        else:
            print ("Will form radial groups before pairings...")
            assert (qrmin is not None)
            assert (qrmax is not None)
            assert (dqr is not None)

        self.qrmin = qrmin
        self.qrmax = qrmax
        self.dqr = dqr
        self.poly_stride = poly_stride
        self.min_grp_size = min_grp_size
        self._make_signal_column(signal)
        if signal is not None:
            self._filter_from_signal( min_signal)

#       exclude exposures with a high/low mean intensity
        if rm_mean_outliers:
            print ("Removing shot mean outliers...")
            self._drop_shotmean_outliers(thresh=rm_mean_thresh)
        
        self.groups = None
        self.u_gruops = None
        if not fixed_qr: 
            self._group_shots_by_peakpos()

    def _make_signal_column(self, signal):
        assert( signal in [None, 'mean', 'stdev', 'snr'] )
#           whether to apply a secondary signal threshhold
        if signal=='mean':
            print("Will use the shot mean filter...")
            self.df['sig'] = self.df.shot_mean 
        elif signal == 'stdev':
            print("Will use the shot standard deviation filter...")
            self.df['sig'] = self.df.shot_stdev
        elif signal== 'snr':
            print ( "Will use signal to noise filter, be sure you used the",\
                    "pk_detect option when making the database or this",\
                    "will break.")
            assert( self.df.pk_amp[0] is not None and \
                        self.df.background_offset is not None )
            self.df['sig'] = self.df.pk_amp / self.df.background_offset

    
    def _filter_from_signal(self, min_signal):
        n_total = len(self.df)
        if min_signal is not None:
            self.df = self.df.query('sig >= %f'%min_signal )
        if max_signal is not None:
            self.df = self.df.query('sig <= %f'%max_signal )
        n_removed = n_total - len(self.df)
        print ("Removed %d/%d exposures from analysis"%(n_removed,
                                                        n_total))
    
    def _drop_shotmean_outliers(self, thresh):
        outliers = helper.is_outlier( self.df.shot_mean.values, thresh )
        where_outliers = np.where(outliers)[0]
        if where_outliers.size:
            n_total = len(self.df)
            self.df.drop( self.df.index[where_outliers], inplace=True)
            n_removed = n_total-len(self.df)
            print ("Removed %d/%d exposures from analysis"%(n_removed,
                                                            n_total))

    def _group_shots_by_peakpos(self):
        """group according to pk_pos parameter, e.g. fitted q111"""
        print ("grouping shots according to radial intensity peak position.")
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
        self.eps_min = []
        if not self.fixed_qr:        
            print ("Iterating over groups and pairing")
            self._iterate_groups() 

        else:
            print ("Pairing all remaining shots in database")
            self.df_g = self.df
            self._pair_group()
            self._find_min_pairs()
        
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
            self._find_min_pairs()

    def _pair_group(self):
        """
        Pairs exposures in a dataframe group according to the
        similarity of their respective polynomials
        """
        ngroup = len(self.df_g)
       
        # polynomial at lower resolution than the ring (it is smooth)
        Px = np.arange(self.nphi)[::self.poly_stride]
        Py = np.vstack( [np.polynomial.chebyshev.chebval(Px,c) 
                            for c in self.df_g.cheby_fit_pkremove])
                            #for c in self.df_g.cheby_fit_compare])
        
        # polynomial distance measure between shot i and shot j
        eps = np.array( [ [sum((Py[i]-p)**2)
                             for p in Py ] 
                          for i in xrange(ngroup)]  )
       
        # Add the max to the diagonal, otherwise the diagonal will 
        # register as the minimum 
        epsI = eps.max(1)*np.identity(ngroup)
        eps += epsI
        
        self.eps_order = np.argsort( eps,1)


    def _find_min_pairs(self):
        pair_inds = []
        for i in xrange(len(self.eps_order) ):
            min_row = list(self.eps_order[i])
            min_pair = [i, min_row.pop(0) ]
            while np.any([ind in pair_inds 
                            for ind in min_pair]):
                try:
                    min_pair = [i, min_row.pop(0)]
                except IndexError:
                    min_pair = None
                    break
            if min_pair is not None and len(set(min_pair)) > 1:
                pair_inds.extend(min_pair)
#               min_pair is indexed from 0 to num of shots in group-outlier. 
#               df_g.tag.keys() are still indexed from 0 to the original number of shots
                min_pair_key=[ self.df_g.tag.keys()[this_pair] for this_pair in min_pair]
                tag_pair = map(str, self.df_g.tag[min_pair_key])
                self.tag_pairs.extend( tag_pair)

    
    def _save_tag_pairs(self):
        """
        saves the tag pairings list in a json file
        """
        self.tag_pairs = zip( self.tag_pairs[::2], 
                                    self.tag_pairs[1::2] )
        print ("Made %d tag pairings..."%len(self.tag_pairs))
        outfile_ = open( self.outfile, 'w')
        json.dump( self.tag_pairs, outfile_)
        outfile_.close()
        print ("Saved tag pairings in json file: %s"%self.outfile)

if __name__ == '__main__':
    db_pickle = 'test.pkl'
    outf = 'test_pkremove.json'    
#   qrmin = 20 # min q radius relative to qmin (pixel units)
#   qrmax = 60 # max q ..                  
#   makeTagPairs = MakeTagPairs(db_pickle, qrmin=qrmin, dqr=3, 
#                        qrmax=qrmax, signal='mean', min_signal=325,
#                        rm_mean_thresh=5, max_signal=3300)
    
#    makeTagPairs.Make(outf)
        
    makeTagPairs = MakeTagPairs(db_pickle, fixed_qr=True,poly_stride=2) 
                                #rm_mean_outliers=True,
                                #rm_mean_thresh=5, poly_stride=2)
    makeTagPairs.Make(outf)
