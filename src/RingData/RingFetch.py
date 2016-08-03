import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy import spatial


class RingFetch:

    def __init__(self, a, b, img_shape=None, img=None, mask=None, q_resolution=0.05,
                 phi_resolution=0.5, wavelen=None, pixsize=None,
                 detdist=None, photon_conversion_factor=1,
                 interp_method='floor', index_query_fname=None):
        '''
        Description
        ===========
        Sums the intensities in a diffraction image to form effective pixel
        readouts at a specified resolution.

        Parameters
        ==========

        `a`, `b` is the horizontal, vertical pixel coordinate where
            forward beam intersects detector

        `img_shape` is shape of two-dimensional diffraction image

        `mask` is a boolean array where False,True represent masked,
            unmasked pixels in an image.

        `q_resolution`, `phi_resolution` are desired resolutions of
            one-dimensional output ring in radial, azimuthal
            dimensions (inverse Angstrom and degree units, respectively)

        `wavelen`, `pixsize`, `detdist` are photon-wavelength, pixel-size
            (assuming square pixels) and sample-to-detector-distance in
            Angstrom, meter, and meter units, respectively.

        `photon_conversion_factor`

        `interp_method`

        `index_query_fname`
        '''

        self.x_center = a
        self.y_center = b
        
        assert( img_shape is not None or img is not None)

        assert(interp_method in ['floor', 'nearest', 'nearest4', 'weighted4'])
        self.method = interp_method
        
        if img is not None:
            self.img_shape = img.shape
            self.set_working_image(img)
        else:
            self.img_shape = img_shape


        self.q_resolution = q_resolution
        self.phi_resolution = phi_resolution

        self.num_phi_nodes = int(360. / self.phi_resolution)

        if self.method in ['nearest', 'nearest4', 'weighted4']:
            assert(index_query_fname is not None)
            query_file = h5py.File(index_query_fname, 'r')
            self._max_radius_in_query_data = max(
                map(int, query_file['nearest/dists'].keys()))
            if self.method == 'nearest':
                self._index_data = query_file['nearest']
            else:
                self._index_data = query_file['nearest4']

        if mask is not None:
            assert(mask.shape == self.img_shape)
            if self.method != 'floor':
                self._mask_flat = mask.ravel()
        self.mask = mask

        self._set_max_ring_radius()
        self.pixsize = pixsize

        if wavelen is not None and detdist is not None:
            self.set_params(wavelen, detdist)

        self.wavelen = wavelen
        self.detdist = detdist
        self.photon_conversion_factor = photon_conversion_factor

    def _set_conversion_functions(self):
        #       convert pixel radius to q
        self.r2q = lambda R: np.sin(np.arctan(
            R * self.pixsize / self.detdist) / 2.) * 4 * np.pi / self.wavelen
#       ... and vice versa
        self.q2r = lambda Q: np.tan( 2 * np.arcsin(Q * self.wavelen / 4 / np.pi)) \
            * self.detdist / self.pixsize

        self.r2theta = lambda R: np.arctan(
            R * self.pixsize / self.detdist) / 2.

    def _set_max_ring_radius(self):
        possible_max_radii = (self.img_shape[1] - self.x_center,
                              self.x_center,
                              self.img_shape[0] - self.y_center,
                              self.y_center)
        self._maximum_allowable_ring_radius = np.min(possible_max_radii)

    def set_working_image(self, img):
        assert (img.shape == self.img_shape)
        self.img = img
        if self.method != 'floor':
            self._img_flat = img.ravel()

    def set_params(self, wavelen, detdist):
        self.wavelen = wavelen
        self.detdist = detdist
        self._set_conversion_functions()

    def update_wavelen(self, wavelen):
        self.wavelen = wavelen
        self._set_conversion_functions()

    def update_detdist(self, detdist):
        self.detdist = detdist
        self._set_conversion_functions()

    def set_photon_factor(self, factor):
        self.photon_conversion_factor = factor

###############
# MAIN METHOD #
###############
    def fetch_a_ring(self, q=None, radius=None, solid_angle=True):
        '''
        Parameters
        ==========

        `q` is moementum transfer magnitude of ring in inverse angstroms
        
        `radius` is radius of ring in pixel units


        `solid_angle` is whether or not to do the solid angle correction

        Returns
        =======

        A tuple of ( `phi_values`, `polar_ring_final`)

        `phi_values` are the azimuthal centers to the effective pixels
            at the desired resolution

        `polar_ring_final` is the corresponding intensities in the
            effective pixels
        '''
        assert(self.wavelen is not None and self.detdist is not None)
        assert( q is not None or radius is not None)
        assert ( None in [ q, radius] )
        self._define_radial_extent_of_ring(q, radius)
        self._check_ring_edges()
        self._make_output_containers()
        self._iterate_across_ring_extent()
        if solid_angle:
            self._solid_angle_correction()
        self._adjust_fractional_rings()
        return self._ring_output.sum(0)

    def _define_radial_extent_of_ring(self, q, radius):
        if radius is None:
            q_of_ring = q
        else:
            q_of_ring = self.r2q(radius)
#       min q for ring at desired resolution
        qmin = q_of_ring - self.q_resolution / 2.
#       max q for ring at desired resolution
        qmax = q_of_ring + self.q_resolution / 2.
#       qmin/qmax in radial pixle units
        self._rmin = self.q2r(qmin)
        self._rmax = self.q2r(qmax)
        assert(self._rmax - 1. > self._rmin)

        self._radii = np.arange(int(self._rmin), int(self._rmax) + 1)
        self._nrad = len(self._radii)

    def _check_ring_edges(self):
        nphi_min = int(2 * np.pi * self._rmin)
        assert(nphi_min >= self.num_phi_nodes)
        assert(int(self._rmax) + 1 < self._maximum_allowable_ring_radius)
        if self.method != 'floor':
            assert(int(self._rmax) + 1 < self._max_radius_in_query_data)

    def _make_output_containers(self):
        self._ring_output = np.zeros((self._nrad, self.num_phi_nodes))

    def _iterate_across_ring_extent(self):
        for self._rad_ind, radius in enumerate(self._radii):
            #           number of azimuthal points in radial image
            self._nphi = int(2 * np.pi * radius)
            if self.method == 'floor':
                self.InterpSimp = InterpSimple(
                    self.x_center,
                    self.y_center,
                    radius + 1,
                    radius,
                    self._nphi,
                    raw_img_shape=self.img.shape)
                self._set_polar_ring()

            elif self.method == 'nearest':
                self._set_polar_ring_mean_nearest(radius)

            elif self.method == 'nearest4':
                self._set_polar_ring_mean_nearest4(radius)

            else:
                self._set_polar_ring_weighted_nearest4(radius)

            self._fill_in_masked_values()
            self._sum_intensity_in_azimuthal_nodes()

    def _solid_angle_correction(self):
        theta_vals = self.r2theta(self._radii)
        solid_angle_per_radii = np.cos(theta_vals) ** 3
        self._ring_output *= solid_angle_per_radii[:, None]

    def _adjust_fractional_rings(self):
        start_factor = 1. - self._rmin + np.floor(self._rmin)
        stop_factor = self._rmax - np.floor(self._rmax)
        self._ring_output[0] *= start_factor
        self._ring_output[-1] *= stop_factor

    def _set_polar_ring_weighted_nearest4(self, radius):
        dists, inds = self._index_data["dists/%d" % radius].value,\
            self._index_data["inds/%d" % radius].value,
        weights = dists / dists.sum(1)[:, None]
        self._polar_ring = np.average(
            self._img_flat[inds], axis=1, weights=weights)
        if self.mask is not None:
            self._polar_ring_mask = np.average(
                self._mask_flat[inds], axis=1, weights=weights)
            self._polar_ring_mask = self._polar_ring_mask.astype(int)

    def _set_polar_ring_mean_nearest4(self, radius):
        inds = self._index_data["inds/%d" % radius].value,
        self._polar_ring = self._img_flat[inds].mean(1)
        if self.mask is not None:
            self._polar_ring_mask = self._mask_flat[inds].mean(1)
            self._polar_ring_mask = self._polar_ring_mask.astype(int)

    def _set_polar_ring_mean_nearest(self, radius):
        inds = self._index_data["inds/%d" % radius].value,
        self._polar_ring = self._img_flat[inds]
        if self.mask is not None:
            self._polar_ring_mask = self._mask_flat[inds]

################################
# RING RADII ITERATION METHODS #
################################
    def _set_polar_ring(self):
        polar_img = self.InterpSimp.nearest(self.img)
        self._polar_ring = polar_img[0]

    def _fill_in_masked_values(self):
        if self.mask is not None and self.method == 'floor':
            #           interpolate the polar mask
            polar_mask = self.InterpSimp.nearest(self.mask.astype(int))
            self._polar_ring_mask = polar_mask[0]

        self._fill_polar_ring()

    def _sum_intensity_in_azimuthal_nodes(self):
        node_edges = np.linspace(
            0,
            self._nphi,
            self.num_phi_nodes +
            1)  # fractional pixels
        node_inds = node_edges.astype(int)
        frac_start = (1 - node_edges + np.floor(node_edges))[:-1]
        frac_stop = (node_edges - np.floor(node_edges))[1:]
        for i in xrange(self.num_phi_nodes - 1):
            start = node_inds[i]
            stop = node_inds[i + 1]
            summed_intens = self._polar_ring[start] * frac_start[i] \
                + np.sum(self._polar_ring[start + 1:stop])\
                + self._polar_ring[stop] * frac_stop[i]
            self._ring_output[self._rad_ind, i] += summed_intens \
                * self.photon_conversion_factor
        start = node_inds[-2]
        self._ring_output[self._rad_ind, -1] += (self._polar_ring[start] * frac_start[-1]
                                               + np.sum(self._polar_ring[start + 1:]))\
            * self.photon_conversion_factor

########################################
# METHODS FOR FILLLING IN MASKED REGIONS
########################################
    def _fill_polar_ring(self):
        """sample_width in degrees"""
        self._sample_width = int(
            round(
                self._nphi *
                10 *
                self.phi_resolution /
                360.))
        assert(self._sample_width > 1)
        self._find_gap_indices()
        self._set_left_and_right_edge_indices()
        self._iterate_over_masked_regions()

    def _find_gap_indices(self):
        mask = self._polar_ring_mask.astype(bool)
        mask_rolled_right = np.roll(mask, 1)
        mask_rolled_left = np.roll(mask, -1)

        is_start_of_gap = (mask ^ mask_rolled_left) & mask
        is_end_of_gap = (mask ^ mask_rolled_right) & mask

        self._gap_start_indices = np.where(is_start_of_gap)[0]
        self._gap_end_indices = np.where(is_end_of_gap)[0]

    def _set_left_and_right_edge_indices(self):
        #if self._polar_ring_mask[0] == 0:
        self._gap_index_pairs = zip(self._gap_start_indices,
            np.roll(self._gap_end_indices, -1))
        #else:
        #    self._gap_index_pairs = zip(self._gap_start_indices,
        #                               self._gap_end_indices)

    def _iterate_over_masked_regions(self):
        
        for self._iStart, self._iEnd in self._gap_index_pairs:
            self._set_gap_size_and_ranges()
            self._get_linear_gap_interpolator()
            self._get_edge_noise()
            self._fill_in_masked_region_with_Gaussian_noise()

    def _set_gap_size_and_ranges(self):
        if self._iEnd < self._iStart:
            self._gap_size = self._iEnd + self._nphi - self._iStart
            self._interpolation_range = np.arange(
                self._iStart, self._iEnd + self._nphi)
        else:
            self._gap_size = self._iEnd - self._iStart
            self._interpolation_range = np.arange(self._iStart, self._iEnd)

    def _get_linear_gap_interpolator(self):
        #       estimate the edge means
        left_mean = self._polar_ring[
            self._iStart - self._sample_width:self._iStart].mean()
        right_mean = self._polar_ring[
            self._iEnd +
            1:self._sample_width +
            self._iEnd +
            1].mean()

#       find the equation of line connecting edge means
        slope = (right_mean - left_mean) / self._gap_size
        self._line = lambda x: slope * (x - self._iStart) + left_mean

    def _get_edge_noise(self):
        #       estimate the edge noise
        left_dev = self._polar_ring[
            self._iStart - self._sample_width:self._iStart].std()
        right_dev = self._polar_ring[
            self._iEnd +
            1:self._sample_width +
            self._iEnd +
            1].std()
        self._edge_noise = (left_dev + right_dev) / 2.  # np.sqrt(2)

    def _fill_in_masked_region_with_Gaussian_noise(self):
        #       fill in noise about the line
        gap_range = self._interpolation_range % self._nphi
        gap_vals = np.random.normal(self._line(self._interpolation_range),
                                    self._edge_noise)
        self._polar_ring[gap_range] = gap_vals


class InterpSimple:

    def __init__(self, a, b, qRmax, qRmin, nphi, raw_img_shape, bin_fac=None):
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
            self.Y, self.X = zoom(np.random.random(raw_img_shape),
                                  1. / bin_fac, order=1).shape
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

    def nearest(self, data_img, dtype=np.float32):
        '''return a 2d np.array polar image'''
        if self.y_centerin_fac:
            data_img = zoom(data_img, 1. / self.y_centerin_fac, order=1)
        data = data_img.ravel()
        return data[self.indices_1d]


class PolarTree:
    def __init__(self, a, b, img_shape, offset_pix=False):
        self.img_shape = img_shape
        self._define_polar_indices(a, b, offset_pix)
        self._create_the_tree()

    def _define_polar_indices(self, a, b, offset_pix):
        Y, X = np.indices(self.img_shape)
        if offset_pix:
            a += 0.5
            b += 0.5
            X += 0.5
            Y += 0.5
        R = np.sqrt((X - a)**2 + (Y - b)**2)
        # for some reason this is a heck of a lot faster when R is rounded..
        R = np.round(R, 0)
        # I am unsure of the consequences, probable some rounding errors...
        PHI = np.arctan2(Y - b, X - a)
        self._points = zip(R.ravel(), PHI.ravel())

    def _create_the_tree(self):
        print ("Making the K-D tree...")
        self.tree = spatial.cKDTree(self._points)

    def set_working_image(self, img):
        assert(img.shape == self.img_shape)
        self.img_flat = img.ravel()

    def mean_nearest_k(self, points, k):
        inds = self.tree.query(points, k=k)[1]
        return self.img_flat[inds].mean(1)

    def weighted_nearest_k(self, points, k):
        weights, inds = self.tree.query(points, k=k)
        weights /= dists.sum(1)[:, None]
        return np.average(self.img[inds], axis=1, weights=weights)
