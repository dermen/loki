#AUTHOR: dermen dermendarko@gmail.com

import sys
import os
import re

import json
import h5py
import pandas
import numpy as np
from scipy import optimize
from scipy.signal import argrelextrema

from loki.utils import postproc_helper as helper

log_ret = '\x1b[80D\x1b[1A\x1b[K'


class MakeDatabase:
    def __init__(self, data_fname, run, 
                    pk_radius=None, pk_detect=True):
        """
        'data_fname'    is a file path which contains 
                        output from process_sacla.interpolate_run
        
        'run'           string indicating the experimental run

        'pk_radius'        is where to look on each 
                        interpolated image in q to parametrize 
                        the exposures using polynomial fits, 
                        in pixel units. Default is None, 
                        in which case peak detection is performed.
        
        'pk_detect'     is whether or not to check for peaks in 
                        each exposure and manually set pk_pos per shot. 
                        If True, then pk_pos is ignored
        """
        
        self.pk_radius = pk_radius
        if self.pk_radius is not None:
            assert( not pk_detect )
        
        assert( os.path.exists( data_fname) )
        print ("\nWill load interpolation data from %s.."%data_fname)
        print ("Will use %s as the run tag."%run)

        self.run = run
        self.data_fname = data_fname
        
        self.pk_detect = pk_detect
        if not self.pk_detect:
            assert(self.pk_radius is not None)
            print ("will not detect peaks, instead will use qRadius %d as"%pk_radius,\
                    "\n position for fitting polynomials.")
        else:
            print ("Will be using peak detection... to fit polynomials")
        
        self._load_data_file()
        self._init_lists()

    def _load_data_file(self):
        """ load data from process_sacla.interpolate_run
        hdf5 file"""
        data = h5py.File( self.data_fname, 'r')
        self.rings = data['ring_intensities']
        self.nshots = len(self.rings)
        self.pmask = data['polar_mask'].value
        self.pmask_bool = np.logical_not( self.pmask.astype(bool) )
        self.qs = data[ 'q_mapping'].value
        self.radii = data['q_radii'].value
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
        

        if self.pk_radius is not None:
            assert( self.pk_radius in self.radii )
            self.pk_pos = np.where( self.radii == self.pk_radius)[0][0]
        else:
            self.pk_pos = None
        

        print ("Data loaded from file %s"%self.data_fname)
        print (" qmin=%f, qmax=%f, num_qs=%d"%(self.qs[0], 
                                            self.qs[-1], 
                                            len(self.qs)))
        print (" total number of shots: %d"%self.nshots)

    
    def _init_lists(self):
        """Initialize some lists"""
        print ("Initializing database list containers...")
        # shot ID lists (indices and tads of the shots)
        self.all_shot_tag = []
        
        # shot global parameters e.g. mean intensity
        self.all_shot_rad_pro = []
        self.all_shot_mean = []
        self.all_shot_stdev = []
        self.all_shot_sum = []
        


#       if detecting Bragg rings on shot by shot basis
        if self.pk_detect:
    
            self.all_pk_inds = [] # each detected peak
            self.all_pk_amp = [] # fitted amplitude of peak
            self.all_pk_offset = [] # fitted offset of peak
            self.all_pk_width = [] # fitted width of peak
            self.all_pk_pos = [] # maximum paek from pk_inds
        else:
            self.all_pk_inds = None # each detected peak
            self.all_pk_amp = None # fitted amplitude of peak
            self.all_pk_offset = None # fitted offset of peak
            self.all_pk_width = None # fitted width of peak
            self.all_pk_pos = self.pk_pos # maximum paek from pk_inds

        # for storing the polynomial coefficients
        # for removing peaks prior to correlating
        self.all_chebyfit_pkremove = [] 
        # for comparing exposures
        self.all_chebyfit_compare = [] 
       
    def Make(self, save_name, pdeg=15, pk_fit_extent=20, pk_fit_width=10,
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
        self.pdeg = pdeg
        self.pk_fit_extent = pk_fit_extent
        self.pk_fit_width = pk_fit_width
        self.remove_spots = remove_spots
        
        if self.remove_spots:
            print( "Will remove bright spots on each angular profile.")
        else:
            print ("Will not remove bright spots on angular profile.")

        self.spot_thresh = spot_thresh
        self.spot_extent = spot_extent
        
        # making database
        self._iterate_shots()
        self._make_datadict()
        self._make_dataframe()
        self._save_dataframe(save_name)

    def _iterate_shots(self):
        """ iterate over all shots in self.rings """
        for shot_ind, shot_tag in enumerate(self.rings):
            print "%sParameterizing exposures ( %d/%d )"%(log_ret,
                                            shot_ind+1,self.nshots)

            shot_rings = self.rings[shot_tag].value
            
            self.all_shot_tag.append(shot_tag)
           
            # the masked polar image
            self.shot_ma = np.ma.masked_array(data=shot_rings, 
                                            mask=self.pmask_bool)
            
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
#       all quantities take into account the mask
        
        self.all_shot_mean.append(self.shot_ma.mean())
        self.all_shot_stdev.append(self.shot_ma.std())
        self.all_shot_sum.append(self.shot_ma.sum())
   
    def _detect_peaks_in_rad_pro(self):
        """ 
        detect bragg ring positions and fit Gaussian to 
        the strongest detected peak
        """
        # find the Bragg ring positions
        pk_inds = self._find_peaks()
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
        # interpolate the angular profile from the polar image
        norm, ring, mask = helper.get_ring(
                    pdata=self.shot_ma.data, 
                    pmask=self.pmask,
                    rm_peaks=False, 
                    iq=self.pk_pos)
        
        if self.remove_spots:
            # find polynomial coefficients for the ring
            ofit_tmp, _ , _ = helper.fit_periodic(
                                    ring.copy(), 
                                    mask.copy(), 
                                    deg=self.pdeg)
            
            # remove large bragg spots from the ring profile
            # this function uses the ofit_tmp to subtract a
            # polynomial prior to spot detection, but does not
            # use the polynomial to alter the ring itself
            ring, mask,_ = helper.remove_peaks( 
                                    ring, 
                                    mask, 
                                    thick=self.spot_extent, 
                                    coef=ofit_tmp, 
                                    peak_thresh=self.spot_thresh)
        
        # fit a new polynomial to the ring
        # these will be used for pk removal prior to
        # correlating
        ofit_pk_rm , _,_ = helper.fit_periodic(
                                ring, 
                                mask, 
                                deg=self.pdeg)
        
        # fit a polynomial to the normalized ring 
        # These will be used to compare exposures
        norm = ring[mask > 0].mean()
        ofit_compare, _ , _ = helper.fit_periodic(
                                ring/norm, 
                                mask, 
                                deg=self.pdeg)
        
        # store the polynomial coefficients
        self.all_chebyfit_pkremove.append(ofit_pk_rm.tolist())
        self.all_chebyfit_compare.append(ofit_compare.tolist())
    
    def _poly_save_dummie(self):
        self.all_chebyfit_pkremove.append([])
        self.all_chebyfit_compare.append([])
    
    def _make_datadict(self):
        print ("Making the database hash table...")
        self.df_dict = {'tag':self.all_shot_tag,#
                        'pk_inds':self.all_pk_inds,#
                        'pk_pos':self.all_pk_pos,#
                        'pk_width':self.all_pk_width,
                        'ring_mean_intensities':self.all_shot_rad_pro , #
                        'pk_amp':self.all_pk_amp,#
                        'background_offset':self.all_pk_offset,#
                        'shot_mean':self.all_shot_mean,#
                        'shot_stdev':self.all_shot_stdev,#
                        'shot_sum':self.all_shot_sum, #
                        'cheby_fit_pkremove':self.all_chebyfit_pkremove, #
                        'cheby_fit_compare':self.all_chebyfit_compare} #

    def _make_dataframe(self):
        print ("Making the database object...")
        self.df = pandas.DataFrame(self.df_dict)
#       Add meta data like a boss because I can
        self.df['run'] = self.run
        self.df['data_filename'] = self.data_fname
        self.df['wavelen'] = self.wavelen
        self.df['pixsize'] = self.pixsize
        self.df['detdist'] = self.detdist
        self.df['detgain'] = self.detgain
        self.df['photon_factor'] = self.photon_factor
        self.df['phi_resolution'] = self.phi_res 
        self.df['q_resolution'] = self.q_res
        self.df['num_phi'] = self.num_phi
        self.df['x_center'] = self.x_center
        self.df['y_center'] = self.y_center
        self.df['radii'] = [self.radii for i in xrange(self.nshots)] 

    def _save_dataframe(self, db_fname):
        self.df.to_pickle(db_fname)
        print ("Database saved as a pandas pickle: %s\n"%db_fname)

if __name__=='__main__':
    pdata_f = "179210_rings_5.hdf5"
    output_f = "test.pkl" 
    run = '179210'
    pk_radius = 695

    makeDB = MakeDatabase(pdata_f, 
                        run=run, 
                        pk_radius=pk_radius, pk_detect=False)
    
    makeDB.Make(save_name=output_f, remove_spots=False)

#   I haven't tried yet, but usage for DNA data would be the following
 
    #pk_pos = 10 # in pixel units relative to min q in polar img
    #makeDB = MakeDatabase(pdata_f, 
    #                    map_f, 
    #                    run=run, 
    #                    pk_detect=False, 
    #                    pk_pos=pk_pos)
    #makeDB.Make(save_name=output_f, 
    #               remove_spots=False)

