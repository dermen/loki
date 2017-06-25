import h5py
import numpy as np
from scipy.ndimage import zoom
import numpy.ma as ma
from itertools import izip



class InterpSimple:

    def __init__(self, a, b, 
        qRmax, qRmin, 
        nphi, raw_img_shape, 
        bin_fac=None:
        '''
        Define a polar image with dimensions: (qRmax-qRmin) x nphi
        ==========================================================
        a,b           - float, x_center,y_center of cartesian image that is
                                being interpolated in pixel units
        qRmin,qRmax     - cartesianimage will be interpolated from
                            qR-qRmin, qR + qRmax in pixel units
        nphi          - int, azimuthal dimension of polar image
                            (number of azimuthal points along polar image)
        raw_img_shape - int tuple, ydim, xdim of the raw image
        bin_fac       - float, reduce the size of the cartesian image
                            by this amount
                        CAUTION: if bin_fac is not None, then a, b passed
                        to this class need to be center fitted from images
                        binned by the same bin_fac. Same goes with qRmax,
                        qRmin

        '''
        self.x_center = a
        self.y_center = b
        self.qRmin = qRmin
        self.qRmax = qRmax
        self.raw_shape = raw_img_shape
        self.phis_ring = np.arange(nphi) * 2 * np.pi / nphi
        self.num_phis_ring = nphi

        self.y_centerin_fac = bin_fac

        if bin_fac:
	    self.Y = int(raw_img_shape[0]/bin_fac)+bool(raw_img_shape[0]%bin_fac)
	    self.X = int(raw_img_shape[1]/bin_fac)+bool(raw_img_shape[1]%bin_fac)

        else:
            self.X = raw_img_shape[1]  # fast dimension
            self.Y = raw_img_shape[0]  # slow dimension

        self.Q = np.vstack([np.ones(self.num_phis_ring) * iq
                            for iq in np.arange(self.qRmin, self.qRmax)])
        self.PHI = np.vstack([self.phis_ring
                              for iq in np.arange(self.qRmin, self.qRmax)])
        self.xring = self.Q * np.cos(self.PHI - np.pi) + self.x_center
        self.yring = self.Q * np.sin(self.PHI - np.pi) + self.y_center

        self.xring_near = self.xring.astype(int) + \
            np.round(self.xring -
                     np.floor(self.xring)).astype(int)
        self.yring_near = self.yring.astype(int) + \
            np.round(self.yring -
                     np.floor(self.yring)).astype(int)
#       'C' style ordering
        self.indices_1d = self.X * self.yring_near + self.xring_near

    def nearest(self, data_img, 
        dtype=np.float32, mask = None):
        '''return a 2d np.array polar image'''

        data = data_img.ravel()
        return data[self.indices_1d]
    

    def set_polar_tree( self, index_query_fname, weighted=True): 
        #self.PT = PolarTree(self.x_center, self.y_center, (self.Y, self.X), offset_pix=offset_pix)
        #phi_range = np.linspace(-np.pi, np.pi, self.num_phis_ring)
        #ring_pts = [ zip([r]*self.num_phis_ring, phi_range) 
        #    for r in np.arange(self.qRmin, self.qRmax)]
        #self._dists, self._inds =  zip(*[ PT.tree.query(rps, k=4) for rps in ring_pts])
        
        qf = h5py.File(index_query_fname, 'r')
        if weighted:
            grp = qf['nearest4']
            self.weighted = True 
        else:
            self.weighted = False
            grp = qf['nearest']
        self._inds =  [ grp['inds'][r].value   
            for r in map( str,np.arange(self.qRmin,self.qRmax))]
        self._dists =  [ grp['dists'][r].value   
            for r in map( str,np.arange(self.qRmin,self.qRmax))]

    def nearest_query(self, data_img, dtype=np.float32, weighted=True):
        data = data_img.ravel()
        if self.weighted:
            rings = np.array( [ np.average(data[i], axis=1, weights=d/d.sum(1)[:,None]) 
                for i,d in izip(self._inds, self._dists) ] )
        else:
            rings = np.array( [ data[i] for i in self._inds ] )
        return rings
