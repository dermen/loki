import sys
import os
import re

import json
import h5py
import pandas
import numpy as np
from scipy import optimize
from scipy.signal import argrelextrema

import postproc_helper as helper

log_ret = '\x1b[80D\x1b[1A\x1b[K'

class MakeDatabase:
    def __init__(self, data_fname, tag_map_fname, run, 
                    pk_pos=None, pk_detect=True):
        """
        'data_fname' is a file path which contains 
            output from process_sacla.interpolate_run
        
        'tag_map_fname' is a json file output from 
            process_sacla.interpolate_run
        
        'run' string indicating the experimental run

        'pk_pos' is where to look on each 
            interpolated image in q to parametrize 
            the exposures using polynomial fits, 
            in relative pixel units (e.g. 0 corresponds 
            to qmin in polar image). Default is None, 
            in which case peak detection is performed.
        
        'pk_detect' is whether or not to check for peaks 
            each exposure and manually set pk_pos. 
            If True, then pk_pos is ignored
        """
        self.pk_pos = pk_pos
        
        assert( os.path.exists( data_fname) )
        assert( os.path.exists( tag_map_fname) )
        print "\nWill load interpolation data from %s.."%data_fname
        print "Will load tag mapping from  %s.."%tag_map_fname
        print "Will use %s as the run tag"%run

        self.run = run
        
        self.data_fname = data_fname
        self.tag_map_fname = tag_map_fname
        self.pk_detect = pk_detect
        if not self.pk_detect:
            assert(pk_pos is not None)
            print "will not detect peaks, instead will use %d as\
                    relative peak position."%pk_pos
        else:
            print "Will be using peak detection..."
        self._load_data_file()
        self._load_tag_map()
        self._init_lists()

    def _load_data_file(self):
        """ load data from process_sacla.interpolate_run
        hdf5 file"""
        data = h5py.File( self.data_fname, 'r')
        self.pd = data['polar_data']
        self.pmask = data['polar_mask'].value
        self.nphi = data['num_phi'].value
        self.qs = data[ 'q_mapping'][:,1]
        self.wavelen_exp = data['wavelen'].value
        self.pixsize_exp = data['pixsize'].value
        self.detdist_exp = data['detdist'].value
        print "Data loaded from file %s"%self.data_fname
        print " wavelength: %f"%self.wavelen_exp 
        print " pixsize: %f"%self.pixsize_exp 
        print " detdist: %f"%self.detdist_exp 
        print " qmin=%f, qmax=%f, num_qs=%d"%(self.qs[0], 
                                            self.qs[-1], 
                                            len(self.qs))
        print " total number of shots: %d"%self.pd.shape[0]


    def _load_tag_map(self):
        """
        load info from process_sacla.interpolate_run
        json file
        """
        # tag:index pairs corresponding to self.pd
        self.tags_inds = json.load(open(self.tag_map_fname, 'r'))
        # switch the keys/tags
        self.inds_tags = { ind:tag
                for tag,ind in self.tags_inds.iteritems() } 
        print "Loaded tag mapping from %s."%self.tag_map_fname

    def _init_lists(self):
        """Initialize some lists"""
        print "Initializing database list containers..."
        # shot ID lists (indices and tads of the shots)
        self.all_shot_index = []
        self.all_shot_tag = []
        
        # shot global parameters e.g. mean intensity
        self.all_shot_rad_pro = []
        self.all_shot_mean = []
        self.all_shot_stdev = []
        self.all_shot_sum = []
        
        # if detecting Bragg rings on shot by shot basis
        self.all_pk_inds = [] # each detected peak
        self.all_pk_amp = [] # fitted amplitude of peak
        self.all_pk_offset = [] # fitted offset of peak
        self.all_pk_width = [] # fitted width of peak
        self.all_pk_pos = [] # maximum paek from pk_inds
         
        # for storing the polynomial coefficients
        # for removing peaks prior to correlating
        self.all_chebyfit_pkremove = [] 
        # for comparing exposures
        self.all_chebyfit_compare = [] 
       
    def Make(self, save_name, pk_fit_extent=20, pk_fit_width=10,
                remove_spots=True, spot_thresh=2, spot_extent=10):
        """ 
        make the database and save as a pandas pickle
        'save_name' database file name
        
        'pk_fit_width' estimated width of Bragg peak (pixel units) 
        'pk_fit_extent' fit range near Bragg peak (pixel units)
        'remove_spots' whether to remove bright bragg peaks
                        or outlier pixels from a given exposure
        'spot_thresh' threshold for detecting spots along the
                    angular profile I(phi). in units of median[I(phi)]
        'spot_extent' the width of the spot to remove in pixel units
        """
        
        self.pk_fit_extent = pk_fit_extent
        self.pk_fit_width = pk_fit_width
        self.remove_spots = remove_spots
        if self.remove_spots:
            print "Will remove bright spots on each angular profile."
        else:
            print "Will not remove bright spots on angular profile."

        self.spot_thresh = spot_thresh
        self.spot_extent = spot_extent
        
        # making database
        self._iterate_shots()
        self._make_datadict()
        self._make_dataframe()
        self._save_dataframe(save_name)

    def _iterate_shots(self):
        """ iterate over all shots in self.pd """
        nshots = self.pd.shape[0]
        for shot_ind,shot in enumerate(self.pd):
            print "%sParameterizing exposures ( %d/%d )"%(log_ret,
                                            shot_ind,nshots)

            self.all_shot_index.append(shot_ind)
            self.all_shot_tag.append(self.inds_tags[shot_ind])
           
            # the masked polar image 
            self.shot_ma = np.ma.masked_equal(shot*self.pmask,0)
            self._get_shot_global_params()

            # the radial profile of the polar image
            self.rad_pro = self.shot_ma.mean(1)
            self.all_shot_rad_pro.append(self.rad_pro) 

            if self.pk_detect:
                # returns a bool
                pks_detected = self._detect_peaks_in_rad_pro()
            
            # fit polynomials
            if self.pk_detect and not pks_detected:
                # save empty arrays (no peaks detected= no data)
                self._poly_save_dummie()
            else:
                self._poly_fitting()
 
    def _get_shot_global_params(self):
        """ 
        store some global shot parameters using a masked
        array
        """
        #print "  Storing global parameters..."
#       all quantities take into account the mask
        self.all_shot_mean.append(self.shot_ma.mean())
        self.all_shot_stdev.append(self.shot_ma.std())
        self.all_shot_sum.append(self.shot_ma.sum())
   
    def _detect_peaks_in_rad_pro(self):
        """ 
        detect bragg ring positions and fit Gaussian to 
        the strongest detected peak
        """
        #print " Beginning the peak detection..."
        # find the Bragg ring positions
        pk_inds = self._find_peaks()
        #print "  Found %d peak(s) in radial profile"%pk_inds.size
        # set initial values
        pk_amp = None
        pk_width = None
        pk_offset = None
        pk_highest = None

        if pk_inds.size > 0:
            pk_magnitudes = self.rad_pro[pk_inds]
            pk_highest = pk_inds[np.argmax(pk_magnitudes)]
            self.pk_pos = pk_highest
            # fit to the highest magnitude  Bragg ring
            pk_params = self._fit_gauss_peak()
            if pk_params is not None:
                pk_amp, pk_offset, pk_var = pk_params[0]
                pk_width = np.sqrt(pk_var)

        # store the peak parameters 
        self.all_pk_inds.append(list(pk_inds))
        self.all_pk_amp.append(pk_amp)
        self.all_pk_width.append(pk_width)
        self.all_pk_offset.append(pk_offset)
        self.all_pk_pos.append(self.pk_pos)

        if pk_inds.size == 0:
            return False
        else:
            return True
            
    def _find_peaks(self):
        """ 
        locate the peaks in I(q); this is necessary because 
        in Feb2014 at SACLA we noticed many exposures with 
        multiple Bragg rings near the q of interest (q111)
        """
        # smooth the I(q) profile
        rad_pro_sm = helper.smooth(self.rad_pro,10,10)
        # simple search for local maxima
        peak_inds = argrelextrema(rad_pro_sm, np.greater)[0]
        return peak_inds
    
    def _fit_gauss_peak(self):
        """ 
        fit a Gaussian to the Bragg Ring profile in I(q) 
        """
        #print " Fitting a Gaussian to brightest peak"
        # extent of the radial peak profile (the peak in I(q) )
        x1 = self.pk_pos - self.pk_fit_extent
        x2 = self.pk_pos + self.pk_fit_extent
        xdata = np.arange(x1,x2)
        # careful near the boundaries of the polar img 
        xdata = xdata[ xdata >= 0 ] 
        xdata = xdata[ xdata < self.rad_pro.shape[0] ] 
        ydata = self.rad_pro[xdata] 
        # compute a gaussian fit, fixing the position of the peak
        fit_param = helper.fit_gauss_offset_fixed_mu(
                                ydata, 
                                xdata,
                                mu=self.pk_pos,
                                var_guess=self.pk_fit_width**2)
        return fit_param

    def _poly_fitting(self):
        """ 
        fit polynomials to the ring profile I(q=pk_pos, phi)
        """
        #print " Doing the polynomial fitting..."
        # interpolate the angular profile from the polar image
        norm, ring, mask = helper.get_ring(
                    pdata=self.shot_ma.data, 
                    pmask=self.pmask, 
                    iq=self.pk_pos)
        
        if self.remove_spots:
            # find polynomial coefficients for the ring
            ofit_tmp, _ , _ = helper.fit_periodic(
                                    ring.copy(), 
                                    mask.copy(), 
                                    deg=15)
            
            # remove large bragg spots from the ring profile
            # this function uses the ofit_tmp to subtract a
            # polynomial prior to spot detection, but does not
            # use the polynomial to alter the ring itself
            ring, mask,_ = helper.remove_peaks( 
                                    ring, 
                                    mask, 
                                    thick=10, 
                                    coef=ofit_tmp, 
                                    peak_thresh=2)
        
        # fit a new polynomial to the ring
        # these will be used for pk removal prior to
        # correlating
        ofit_pk_rm , _,_ = helper.fit_periodic(
                                ring, 
                                mask, 
                                deg=15)
        
        # fit a polynomial to the normalized ring 
        # These will be used to compare exposures
        norm = ring[mask > 0].mean()
        ofit_compare, _ , _ = helper.fit_periodic(
                                ring/norm, 
                                mask, 
                                deg=15)
        
        # store the polynomial coefficients
        self.all_chebyfit_pkremove.append(ofit_pk_rm.tolist())
        self.all_chebyfit_compare.append(ofit_compare.tolist())
    
    def _poly_save_dummie(self):
        self.all_chebyfit_pkremove.append([])
        self.all_chebyfit_compare.append([])
    
    def _make_datadict(self):
        print "Making the database hash table..."
        self.df_dict = {'hdf5_index':self.all_shot_index,#
                        'tag':self.all_shot_tag,#
                        'pk_inds':self.all_pk_inds,#
                        'pk_pos':self.all_pk_pos,#
                        'pk_width':self.all_pk_width,
                        'radial_profile':self.all_shot_rad_pro , #
                        'pk_amp':self.all_pk_amp,#
                        'background_offset':self.all_pk_offset,#
                        'shot_mean':self.all_shot_mean,#
                        'shot_stdev':self.all_shot_stdev,#
                        'shot_sum':self.all_shot_sum, #
                        'cheby_fit_pkremove':self.all_chebyfit_pkremove, #
                        'cheby_fit_compare':self.all_chebyfit_compare} #

    def _make_dataframe(self):
        print "Making the database object..."
        self.df = pandas.DataFrame(self.df_dict)
        self.df['data_filename'] = self.data_fname
        self.df['wavelen'] = self.wavelen_exp
        self.df['pixsize'] = self.pixsize_exp
        self.df['run'] = self.run

    def _save_dataframe(self, db_fname):
        self.df.to_pickle(db_fname)
        print "Database saved as a pandas pickle: %s"%db_fname
#dermen

if __name__=='__main__':
    pdata_f = "/data/sacla_gold_Feb2014/interped_178802.hdf5"
    map_f = "/data/work/mender/interped_178802.json"
    output_f = "/data/work/mender/interped_178802.pkl" 
    
#   run number from file name
    run_str = re.findall( '_[0-9]{6}', pdata_f )[0]
    run = run_str.split('_')[1]
    
    makeDB = MakeDatabase(pdata_f, 
                        map_f, 
                        run=run)
    
    makeDB.Make(save_name=output_f)

#   I haven't tried yet, but usage for DNA data would be the following
 
    #pk_pos = 10 # in pixel units relative to min q in polar img
    #makeDB = MakeDatabase(pdata_f, 
    #                    map_f, 
    #                    run=run, 
    #                    pk_detect=False, 
    #                    pk_pos=pk_pos)
    #makeDB.Make(save_name=output_f, 
    #               remove_spots=False)


