import sys
import os

import json
import h5py
import pandas
import numpy as np

from loki.utils import postproc_helper as helper
from loki import RingData

log_ret = '\x1b[80D\x1b[1A\x1b[K'

class CorTagPairs:
    def __init__(self, data_fname, tag_pairs_fname=None, 
                    tag_map_fname=None, db_fname=None):
        """
        data_fname, str, the hdf5 file w the polar data
        tag_pairs_fname, a file with tag pairings
        tag_map_fname, a file mapping the tag string to the index 
                        in the hdf5 file
        db_fname, str, a pandas dataframe pickle file
        """
        self.data_fname = data_fname 
        
        self.tag_fname = tag_pairs_fname
        self.map_fname = tag_map_fname
        self.db_fname = db_fname

        self.dif_imgs = None
        self.dif_cors = None
        
        print  ( "\nLoading input files..." )
        self._load_data_file()
        self._load_tagPairs_tagMap_and_database()


    def _load_data_file(self):
        print ( "Loading polar data file %s"%self.data_fname )
        f = h5py.File(self.data_fname, 'r')
        self.pd = f['polar_data']
        self.nq = f['q_mapping'].shape[0]
        self.nphi = f['num_phi'].value
        self.pmask= f['polar_mask'].value

    def _load_tagPairs_tagMap_and_database(self):
        print  ( "Loading tag pairing file %s"%self.tag_fname )
        self.tag_pairs = json.load(open(self.tag_fname, 'r'))
        self.npairs = len(self.tag_pairs)
        print ("Loading index mapping file %s"%self.map_fname )
        self.tag_map = json.load(open( self.map_fname, 'r'))
        print ("Loading database pickle %s"%self.db_fname )
        self.df = pandas.read_pickle(self.db_fname)

    def make_dif_imgs(self, fixed_q, norm=True, del_q=2, iq=None,
                        rm_spots=True, spot_thick=None, spot_thresh=2.5):
        """
        fixed_q - whether or not pk_detect was used when making database
                    if False, then provide iq parameter
        
        norm    -  whether to normalize shots before differencing 
                (highly recommended!)
        del_q   - q thickness of I(phi) in pixel units (will be a average)
        iq     - where to look on each polar image to get I(phi)
                    (in pixel units relative from q min on polar image, such
                      that iq=0 corresponds to qmin
        rm_spots   - whether to remove  bright spots from I(phi)
        
        spot_thick   -  if removing spot, this will remove spot with a 
                   finite width. thick=0/None will remove just the spot,
                   thick=1 will remove the spot and its neighboring pixel,etc
        spot_thresh   - threshhold for spot removal

        """
       
        print( "Making difference angular profiles..." )
        if fixed_q:
            print  ("Will use a fixed q..." )
            assert (iq is not None)
            overbounds1 = iq+del_q+1 >= self.nq 
            overbounds2 = iq-del_q < 0
            assert( not (overbounds1) and not (overbounds2) )
        else:
            print  ("Will use pk_pos to estimate angular profile...") 
        self.dif_imgs = np.zeros( (self.npairs, self.nphi, 2 ) )
        overflow_inds = []
        for i,tags in enumerate(self.tag_pairs):
            print ('%sMaking differencing pair %d/%d'%(log_ret,i,self.npairs) )
            
            t1,t2 = tags
            i1,i2 = self.tag_map[t1], self.tag_map[t2]
           
            if fixed_q:
                iq1 = iq
                iq2 = iq
            else:
                iq1 = self.df.loc[ self.df.tag==t1, 'pk_pos'].values[0]
                iq2 = self.df.loc[ self.df.tag==t2, 'pk_pos'].values[0]
                # overflow conditions
                over1 =iq1+del_q+1 >= self.nq 
                over2 =iq2+del_q+1 >= self.nq
                over3 =iq1-del_q < 0
                over4 =iq2-del_q < 0
                if any( (over1, over2, over3, over4)):
                    overflow_inds.append(i)
                    continue
            
                 
            coef1 = self.df.loc[ self.df.tag==t1, \
                                'cheby_fit_pkremove'].values[0]
            coef2 = self.df.loc[ self.df.tag==t2, \
                                'cheby_fit_pkremove'].values[0]

            r1 = range( iq1-del_q, iq1+del_q+1)
            r2 = range( iq2-del_q, iq2+del_q+1)

            mask1 = np.floor( self.pmask[r1].mean(0))
            mask2 = np.floor( self.pmask[r2].mean(0))
           
            masked = mask1*mask2

            ring1 = mask1* (self.pd[ i1, r1].mean(0))
            ring2 = mask2* (self.pd[ i2, r2].mean(0))

            if rm_spots:
                ring1, mask1, removed1 = helper.remove_peaks(ring1,
                                         mask1, thick=spot_thick, 
                                    coef=coef1, peak_thresh=spot_thresh)
                ring2, mask2, removed2 = helper.remove_peaks( ring2,
                                          mask2, thick=spot_thick, 
                                     coef=coef2, peak_thresh=spot_thresh)
                masked = mask1*mask2

            else:
                masked = mask1*mask2
#           
            ring1 = ring1*masked
            ring2 = ring2*masked

            if norm:
                med1 = np.mean( ring1[ ring1 > 0 ] )
                med2 = np.mean( ring2[ ring2 > 0 ] )
                self.dif_imgs[i,:,0] = (ring1/med1)
                self.dif_imgs[i,:,1] = (ring2/med2)
            else:
                self.dif_imgs[i,:,0] = ring1 
                self.dif_imgs[i,:,1] = ring2 
        
        self.dif_imgs = np.delete(self.dif_imgs, 
                            overflow_inds, axis=0)


    def make_dif_imgs2D(self, fixed_q, norm=True, del_q=2, iq=None,
                rm_spots=True, spot_thick=None, spot_thresh=2.5):
        print("TESTING...")
        print  ("Making difference angular profiles..." )
        if fixed_q:
            print ("Will use a fixed q...")
            assert (iq is not None)
            overbounds1 = iq+del_q+1 >= self.nq 
            overbounds2 = iq-del_q < 0
            assert( not (overbounds1) and not (overbounds2) )
        else:
            print ("Will use pk_pos to estimate angular profile..." )

        n = len( np.arange( -del_q, del_q+1))
        self.dif_imgs = np.zeros( (self.npairs, n, self.nphi, 2 ) )
        overflow_inds = []
        for i,tags in enumerate(self.tag_pairs):
            print ('%sMaking differencing pair %d/%d'%(log_ret,i,self.npairs) )
            
            t1,t2 = tags
            i1,i2 = self.tag_map[t1], self.tag_map[t2]
           
            if fixed_q:
                iq1 = iq
                iq2 = iq
            else:
                iq1 = self.df.loc[ self.df.tag==t1, 'pk_pos'].values[0]
                iq2 = self.df.loc[ self.df.tag==t2, 'pk_pos'].values[0]
                # overflow conditions
                over1 =iq1+del_q+1 >= self.nq 
                over2 =iq2+del_q+1 >= self.nq
                over3 =iq1-del_q < 0
                over4 =iq2-del_q < 0
                if any( (over1, over2, over3, over4)):
                    overflow_inds.append(i)
                    continue
            
                 
            coef1 = self.df.loc[ self.df.tag==t1, \
                                'cheby_fit_pkremove'].values[0]
            coef2 = self.df.loc[ self.df.tag==t2, \
                                'cheby_fit_pkremove'].values[0]

            r1 = range( iq1-del_q, iq1+del_q+1)
            r2 = range( iq2-del_q, iq2+del_q+1)

            mask1 = np.floor( self.pmask[r1].mean(0))
            mask2 = np.floor( self.pmask[r2].mean(0))
           
            masked = mask1*mask2

            ring1 = mask1* (self.pd[ i1, r1].mean(0))
            ring2 = mask2* (self.pd[ i2, r2].mean(0))

            if rm_spots:
                ring1, mask1, removed1 = helper.remove_peaks(ring1,
                                         mask1, thick=spot_thick, 
                                    coef=coef1, peak_thresh=spot_thresh)
                ring2, mask2, removed2 = helper.remove_peaks( ring2,
                                          mask2, thick=spot_thick, 
                                     coef=coef2, peak_thresh=spot_thresh)
                masked = mask1*mask2

            else:
                masked = mask1*mask2
#           
            peak_mask = np.vstack( [masked for _ in r1 ] )
            ring1 = self.pd[i1,r1] * peak_mask*self.pmask[r1]
            ring2 = self.pd[i2,r2]  * peak_mask*self.pmask[r2]

            if norm:
                med1 = np.mean( ring1[ ring1 > 0 ] )
                med2 = np.mean( ring2[ ring2 > 0 ] )
                self.dif_imgs[i,:,:,0] = (ring1/med1)
                self.dif_imgs[i,:,:,1] = (ring2/med2)
            else:
                self.dif_imgs[i,:,:,0] = ring1 
                self.dif_imgs[i,:,:,1] = ring2 

        self.dif_imgs = np.delete(self.dif_imgs, 
                            overflow_inds, axis=0)


    def make_dif_cors(self):
        """ makes difference correlations using RingData.DiffCorr"""
        print ("Making difference correlations...")
        assert( self.dif_imgs is not None)
        differences = self.dif_imgs[:,:,0] - self.dif_imgs[:,:,1]
        DC = RingData.DiffCorr(differences, pre_dif=True)
        self.dif_cors = DC.autocorr()
    
    def make_dif_cors2D(self):
        """ makes difference correlations using RingData.DiffCorr"""
        print ("Making difference correlations...")
        assert( self.dif_imgs is not None)
        differences = self.dif_imgs[:,:,:,0] - self.dif_imgs[:,:,:,1]
        DC = RingData.DiffCorr(differences, pre_dif=True)
        self.dif_cors = DC.autocorr()
    
    def save(self, outfilename):
        """
        outfilename - hdf5 output file name
        """
        print ("Saving output to %s"%outfilename)
        assert( self.dif_cors is not None)
        outf = h5py.File( outfilename, 'w')
        outf.create_dataset('dif_imgs', data=self.dif_imgs)
        outf.create_dataset('tag_pairs', data=map(str,self.tag_pairs))
        outf.create_dataset('dif_cors', data=self.dif_cors)
        outf.close()

if __name__ == '__main__':
    data_f = '/data/sacla_gold_Feb2014/interped_178802.hdf5'
    db_f = '/data/work/mender/interped_178802.pkl'
    pairs_f = '/data/work/mender/interped_178802_pairs.json'
    map_f = '/data/work/mender/interped_178802.json'

    CTP = CorTagPairs(data_fname=data_f,
                        tag_pairs_fname=pairs_f,
                        tag_map_fname =map_f,
                        db_fname=db_f)
    CTP.make_dif_imgs(fixed_q=False, del_q=2)
    CTP.make_dif_cors()
    out_f = '/data/work/mender/correlated_178802.hdf5'
    CTP.save(out_f)
 
