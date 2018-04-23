# AUTHOR: dermen dermendarko@gmail.com
import sys
import os

import h5py
import pandas
import numpy as np

from loki.utils.postproc_helper import fit_periodic, remove_peaks

log_ret = '\x1b[80D\x1b[1A\x1b[K'

class MakeDatabase:

    def __init__(self, data_fname, run, pk_q):
        """
        'data_fname'    is a file path which contains
                        output from process_sacla.interpolate_run

        'run'           string indicating the experimental run

        'pk_q'          is where to look on each
                        interpolated image in q to parametrize
                        the exposures using polynomial fits,
                        in inverse angstroms. 
        """
        
        assert(os.path.exists(data_fname))
        print ("\nWill load interpolation data from %s.." % data_fname)
        print ("Will use %s as the run tag." % run)

        self.data_fname = data_fname
        self.run = run
        self.pk_q = pk_q

        self._load_data_file()

        self._init_lists()

    def _load_data_file(self):
        """ load data from process_sacla.interpolate_run
        hdf5 file"""
        data = h5py.File(self.data_fname, 'r')
        self.rings = data['ring_intensities']
        self.nshots = len(self.rings)
        self.tags = self.rings.keys()
        
        self.radii = data['ring_radii'].value
        self.wavelen = data['wavelen'].value
        self.pixsize = data['pixsize'].value
        self.detdist = data['detdist'].value
        self.detgain = data['detgain'].value
        self.photon_factor = data['photon_factor'].value
        self.phi_res = data['phi_resolution'].value
        self.q_res = data['q_resolution'].value
        self.num_phi = data['num_phi'].value
        self.x_center = data['x_center'].value
        self.y_center = data['y_center'].value
        
        self.how = data['how'].value
        if self.how == 'fetch':
            self.qs = data['ring_moementum_transfer'].value[0]
            self.pk_pos = np.argmin(np.abs(self.qs - self.pk_q))

        else:
            print("Not implemented for floor method. Exiting.")
            sys.exit()

        print ("Data loaded from file %s" % self.data_fname)
        print (" qmin=%f, qmax=%f, num_qs=%d" % (self.qs[0],
                                                 self.qs[-1],
                                                 len(self.qs)))
        print (" total number of shots: %d" % self.nshots)

    def _init_lists(self):
        """Initialize some lists"""
        print ("Initializing database list containers...")
        # shot ID lists (indices and tads of the shots)
        #self.all_shot_tag = []
        # shot global parameters e.g. mean intensity
        self.all_shot_rad_pro = []
        self.all_shot_mean = []
        self.all_shot_stdev = []
        self.all_shot_sum = []
        
        # for storing the polynomial coefficients
        self.all_chebyfit_pkremove = []
        self.all_chebyfit_compare = []

    def Make(self, save_name, pdeg=15,
             remove_spots=True, spot_thresh=5, spot_extent=1):
        """
        make the database and save as a pandas pickle
        'save_name' database file name

        'remove_spots' whether to remove bright bragg peaks
                        or outlier pixels from a given exposure
        'spot_thresh' threshold for detecting spots along the
                    angular profile I(phi). in units of median[I(phi)]
        'spot_extent' the width of the spot to remove in degrees
        """
        self.pdeg = pdeg
        self.remove_spots = remove_spots

        if self.remove_spots:
            print("Will remove bright spots on each angular profile.")
        else:
            print ("Will not remove bright spots on angular profile.")

        self.spot_thresh = spot_thresh
        self.spot_extent =  int( round( spot_extent / (360./ self.num_phi) )) # in pixel units

        # making database
        self._iterate_shots()
        self._make_datadict()
        self._make_dataframe()
        self._save_dataframe(save_name)

    def _iterate_shots(self):
        """ iterate over all shots in self.rings """
        for shot_ind, shot_tag in enumerate(self.tags):

            print ( "%sParameterizing exposure ( %d/%d )" % (log_ret,
                                                            shot_ind + 1, self.nshots) )
            self.shot_rings = self.rings[shot_tag].value
            self.ring = self.shot_rings[self.pk_pos]

            # fit polynomials
            self._poly_fitting()
            
            # the masked polar image
            self.shot_ma = np.ma.masked_array(data=self.ring,
                mask= np.logical_not(self.mask.astype(bool)) )
            
            self._get_shot_global_params()


    def _poly_fitting(self):
        """
        fit polynomials to the ring profile I(q=pk_pos, phi)
        """

        if self.remove_spots:
            # find polynomial coefficients for the ring
            ofit_tmp, _, _ = fit_periodic(
                self.ring.copy(),
                np.ones_like(self.ring),
                deg=self.pdeg)

            # remove large bragg spots from the ring profile
            # this function uses the ofit_tmp to subtract a
            # polynomial prior to spot detection, but does not
            # use the polynomial to alter the ring itself
            ring, self.mask, _ = remove_peaks(
                self.ring,
                np.ones_like(self.ring),
                thick=self.spot_extent,
                coef=ofit_tmp,
                peak_thresh=self.spot_thresh)

        else:
            self.mask = np.ones_like(self.ring)

        # fit a new polynomial to the ring
        # these will be used for pk removal prior to
        # correlating
        ofit_pk_rm, _, _ = fit_periodic(
            self.ring,
            self.mask,
            deg=self.pdeg)

        # fit a polynomial to the normalized ring
        # These will be used to compare exposures
        norm = self.ring[self.mask > 0].mean()
        ofit_compare, _, _ = fit_periodic(
            self.ring / norm,
            self.mask,
            deg=self.pdeg)

        # store the polynomial coefficients
        self.all_chebyfit_pkremove.append(ofit_pk_rm.tolist())
        self.all_chebyfit_compare.append(ofit_compare.tolist())
    
    def _get_shot_global_params(self):
        """
        store some global shot parameters using a masked
        array
        """
#       all quantities take into account the mask
        self.all_shot_mean.append(self.shot_ma.mean())
        self.all_shot_stdev.append(self.shot_ma.std())
        self.all_shot_sum.append(self.shot_ma.sum())
        self.all_shot_rad_pro.append(self.shot_rings.mean(1))

    def _make_datadict(self):
        print ("Making the database hash table...")
        self.df_dict = {'tag': self.tags, 
                        'detdist': self.detdist,
                        'wavelen': self.wavelen,
                        'photon_factor': self.photon_factor,
                        'radial_profile': self.all_shot_rad_pro,
                        'shot_mean': self.all_shot_mean,
                        'shot_stdev': self.all_shot_stdev,
                        'shot_sum': self.all_shot_sum,
                        'cheby_fit_pkremove': self.all_chebyfit_pkremove,
                        'cheby_fit_compare': self.all_chebyfit_compare}

    def _make_dataframe(self):
        print ("Making the database object...")
        self.df = pandas.DataFrame(self.df_dict)
#       Add meta data like a boss because I can
        self.df['pk_pos'] = self.pk_pos
        self.df['run'] = self.run
        self.df['how'] = self.how
        self.df['data_filename'] = self.data_fname
        self.df['pixsize'] = self.pixsize
        self.df['detgain'] = self.detgain
        self.df['phi_resolution'] = self.phi_res
        self.df['q_resolution'] = self.q_res
        self.df['num_phi'] = int( self.num_phi)
        self.df['x_center'] = self.x_center
        self.df['y_center'] = self.y_center

    def _save_dataframe(self, db_fname):
        self.df.to_pickle(db_fname)
        print ("Database saved as a pandas pickle: %s\n" % db_fname)

if __name__ == '__main__':
    pdata_f = 'highQ_rings_179210.hdf5'
    output_f = "test.pkl"
    run = '179210'
    pk_q = 2.99

    makeDB = MakeDatabase(pdata_f,
                          run=run,
                          pk_q=pk_q) 
    makeDB.Make(save_name=output_f)

