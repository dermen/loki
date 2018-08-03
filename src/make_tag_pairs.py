
import h5py
import pandas
import numpy as np
from scipy.spatial import distance

from loki.utils import postproc_helper as helper


class MakeTagPairs:

    def __init__(self, db_pickle, fixed_qr=False, qrmin=None,
                 qrmax=None, dqr=1, rm_mean_outliers=True,
                 rm_mean_thresh=3, min_grp_size=20, signal=None,
                 min_signal=None, max_signal=None):
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

        'signal'       - tells what kind of signal should be thresholded
                        can be ['shot_mean', 'shot_stdev'] 
        'min_signal'   - minimum signal (as determined by the 'signal'
                        parameter)
        'max_signal'   - maximum signal (as determined by the 'signal'
                         parameter)
                        for a shot to be considered for analysis
        """
        self.db_pickle = db_pickle
        print ("\nLoading pandas pickle: %s" % self.db_pickle)
        self.df = pandas.read_pickle(self.db_pickle)
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
        
        self.min_grp_size = min_grp_size
        if signal is not None:
            self._filter_from_signal(signal, min_signal, max_signal)

#       exclude exposures with a high/low mean intensity
        if rm_mean_outliers:
            print ("Removing shot mean outliers...")
            self._drop_shotmean_outliers(thresh=rm_mean_thresh)

        self.df.reset_index(inplace=True)
        self.groups = None
        self.u_gruops = None
        if not fixed_qr:
            self._group_shots_by_peakpos()

    def _filter_from_signal(self, signal, min_signal, max_signal):
        n_total = len(self.df)
        assert( signal in self.df )
        if min_signal is not None:
            self.df = self.df.query('%s >= %f' % (signal, min_signal))
        if max_signal is not None:
            self.df = self.df.query('%s <= %f' % (signal, max_signal))
        n_removed = n_total - len(self.df)
        print ("Removed %d/%d exposures from analysis" % (n_removed,
                                                          n_total))

    def _drop_shotmean_outliers(self, thresh):
        outliers = helper.is_outlier(self.df.shot_mean.values, thresh)
        where_outliers = np.where(outliers)[0]
        if where_outliers.size:
            n_total = len(self.df)
            self.df.drop(self.df.index[where_outliers], inplace=True)
            n_removed = n_total - len(self.df)
            print ("Removed %d/%d exposures from analysis" % (n_removed,
                                                              n_total))

    def _group_shots_by_peakpos(self):
        """group according to pk_pos parameter, e.g. fitted q111"""
        print ("grouping shots according to radial intensity peak position.")
        qbins = np.arange(self.qrmin, self.qrmax, self.dqr)
        self.groups = np.digitize(self.df.pk_pos, bins=qbins)
        self.u_groups = np.unique(self.groups)

    def Make(self, outfilename, poly_thresh=3.5):
        """
        make the tag pairings

        outfilename - name f the hdf5 file which will store the pairings
        poly_thresh - threshhold for detecting outlier chi-square deltas
                        between exposures. If
                                min_s,s'( chi-square)
                        is still too large, then the pair s,s' wont be
                        considered.
        """
        self.outfile = h5py.File(outfilename, 'w')

        self.poly_thresh = poly_thresh
        self.tag_pairs = []
        if not self.fixed_qr:
            print ("Iterating over groups and pairing")
            self._iterate_groups()

        else:
            print ("Pairing all remaining shots in database")
            self.df_g = self.df
            self.groupID = 0
            self._pair_group()

        self.outfile.close()

    def _iterate_groups(self):
        """
        Iterate over groups, determined from _group_shots_by_peakpos,
        pairing each group
        """
        for self.groupID, g in enumerate(self.u_groups):
            self.df_g = self.df.loc[self.groups == g]
            if len(self.df_g) < self.min_grp_size:
                continue
            self.df_g.reset_index(inplace=True)
            self._pair_group()

    def _pair_group(self):
        """
        Pairs exposures in a dataframe group according to the
        similarity of their respective polynomials
        """
        ngroup = len(self.df_g)
        Px = np.arange(self.nphi)
        Py = np.vstack([np.polynomial.chebyshev.chebval(Px, c)
                        for c in self.df_g.cheby_fit_pkremove])

#       polynomial distance measure between shot i and shot j
        print("  Calculating the distance matrix...")
        eps = distance.cdist(Py, Py, metric='euclidean')

        print("  Calculating the preference matrix...")
        # Add the max to the diagonal, otherwise the diagonal will
        # register as the minimum (each shot is closest to itself)
        epsI = 1.1 * eps.max(1) * np.identity(ngroup)
        eps += epsI

#       column 0 is the shot, columns 1->N are the "closest"
#       shots, 1 being the closest and N being the furthest
        self._shot_preference = np.roll(eps.argsort(1), 1, axis=1)

        print("    storing in hash table...")
        pref_dict = {str(E[0]): list(E[1:])
                     for E in self._shot_preference.astype(str)}

#       use the stable roommate (Irving's) algorithm to pair the shots
        print("  Forming the pairings using Irving's algorthm...")
        print("BROKN EXItING , NO PAIRING FOR NOW.... ")
        sys.exit()
        pairs_dict = stable.stableroomate(prefs=pref_dict)

        pairs = self._remove_duplicate_pairs(pairs_dict)

        tag_pairs = [(str(self.df_g.tag[i]), str(self.df_g.tag[j]))
                     for i, j in pairs]

        self.tag_pairs.extend(tag_pairs)

        pdists = [eps[i][j] for i, j in pairs]
        score = np.mean(pdists) / np.mean(eps.ravel())

        print("  Pairing score for group %d: %.3f" % (self.groupID, score))

        self.outfile.create_dataset(
            'group%d/pair_distances' %
            self.groupID, data=pdists)
        self.outfile.create_dataset(
            'group%d/tag_pairs' %
            self.groupID, data=tag_pairs)
        self.outfile.create_dataset('group%d/pairs' % self.groupID, data=pairs)
        self.outfile.create_dataset('group%d/score' % self.groupID, data=score)
        self.outfile.create_dataset(
            'group%d/distance_matrix' %
            self.groupID, data=eps)
        print ("Saved pairing data to %s!"%self.outfile.filename)

    @staticmethod
    def _remove_duplicate_pairs(pairs_dict):
        pairs = []
        for k, v in pairs_dict.iteritems():
            set_p = set((int(k), int(v)))
            if set_p not in pairs:
                pairs.append(set_p)
        return [[i, j] for i, j in pairs]


def pair_shots(eps):
    #epsI = 1.1*eps.max(1) * np.identity(eps.shape[0])
    #eps += epsI
    shot_preference = np.roll(eps.argsort(1), 1, axis=1)
    pref_dict = {str(E[0]): list(E[1:])
                 for E in shot_preference.astype(str)}

    pairs_dict = stable.stableroomate(prefs=pref_dict)
    return pairs_dict


if __name__ == '__main__':
    db_pickle = 'test.pkl'
    outf = 'test_pairs.hdf5'
    makeTagPairs = MakeTagPairs(db_pickle, fixed_qr=True, signal='shot_mean', min_signal=62)
    makeTagPairs.Make(outf)
