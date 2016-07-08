import numpy as np
import h5py
import json
from loki.RingData import InterpSimple, RingFetch

from pylab import plot,show,imshow

#####################################################
# A GROUP OF HELPER FUNCTIONS FOR VARIOUS BEAMTIMES #
#####################################################

def imagesFromTags( runfile ,  tagList):
    """
    runfile,   str, filename of the run hdf5
    tagList,    list str, list of run tags
    """

    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]
    imgs_path     = '/%s/detector_2d_assembled_1'%run_key
    img_gen   = (  fh5[ imgs_path + '/' + tag + '/detector_data' ].value 
                    for tag in tagList )
    return img_gen


def getNorm( runfile, probe=True, pump=False, beam=True):
    
    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]

    monit_path = '/%s/event_info/bl_3/oh_2/bm_2_pulse_energy_in_joule'%run_key

    intens_monit = fh5[monit_path].value

    beam_path = '/%s/event_info/acc/accelerator_status'%run_key
    pump_path ='/%s/event_info/bl_3/lh_1/laser_pulse_selector_status'%run_key
    probe_path = '/%s/event_info/bl_3/eh_1/xfel_pulse_selector_status'%run_key

    beam_stat     = fh5[beam_path].value.astype(bool)
    pump_stat     = fh5[pump_path].value.astype(bool)
    probe_stat    = fh5[probe_path].value.astype(bool)
    
    intens_monit  = [ val for i,val in enumerate( intens_monit) 
                    if beam_stat[i] == beam 
                        and pump_stat[i] == pump 
                        and probe_stat[i] == probe ]

    return np.sum( intens_monit)

def selectImagesSimple( runfile, shutter=1, beam=1, 
                        beamline=3, hutch=1, return_extra=False ):
    """
    Parameters
    ==========
    runfile,   str, filename of the run hdf5
    shutter,   str, 1 for open or 0  for closed
    beam       bool, 1 for beam up or 0 for beam down
    """

    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]
    tags          = fh5['/%s/detector_2d_assembled_1'%run_key].keys()[1:]
    beam_stat     = fh5['/%s/event_info/acc/accelerator_status'%run_key].value
    
    shutt_params = (run_key, beamline, hutch)
    shutt_stat_path = '/%s/event_info/bl_%d/eh_%d/xfel_pulse_selector_status'\
                        %shutt_params
    shutt_stat    = fh5[shutt_stat_path].value

#   keep only the tags corresponding to status
    tags     = [ tag for i,tag in enumerate( tags) 
            if beam_stat[i] == beam and shutt_stat[i] == shutter ]

    img_path = '%s/detector_2d_assembled_1/%s/detector_data'
    img_gen  = (  fh5[img_path%(run_key,tag) ].value 
            for tag in tags )
    
    if return_extra:
        run = run_key.split('_')[1], 
        return img_gen, tags, run, shutt_stat
    else:
        return img_gen, tags

def selectImages(  runfile, probe=True, pump=False, beam=True ):
    """
    Parameters
    ==========
    runfile,   str, filename of the run hdf5
    probe,     bool, if xfel shutter is open or closed
    pump,      bool, if laser shutter is open or closed
    beam       bool, if xfel beam is on
    """
    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]
    tags          = fh5['/%s/detector_2d_assembled_1'%run_key].keys()[1:]
    
    
    beam_path = '/%s/event_info/acc/accelerator_status'%run_key
    pump_path ='/%s/event_info/bl_3/lh_1/laser_pulse_selector_status'%run_key
    probe_path = '/%s/event_info/bl_3/eh_1/xfel_pulse_selector_status'%run_key

    beam_stat     = fh5[beam_path].value.astype(bool)
    pump_stat     = fh5[pump_path].value.astype(bool)
    probe_stat    = fh5[probe_path].value.astype(bool)

#   keep only the tags corresponding to status
    tags         = [ tag for i,tag in enumerate( tags) 
                    if beam_stat[i] == beam 
                        and pump_stat[i] == pump 
                        and probe_stat[i] == probe ]
    
    img_path ='%s/detector_2d_assembled_1/%s/detector_data'
    img_gen   = (  fh5[ img_path%(run_key,tag) ].value 
                    for tag in tags )
    num_im = len(tags)
    return img_gen, tags

def aveImages(imggen, num_img):
    im = imggen.next()
    for im_next in imggen:
        im += im_next
    im /= num_img
    return im

def normalize_polar_images( imgs, mask_val = -1 ): 
    norms = np.ma.masked_equal( imgs, mask_val).mean(axis=2)
    imgs /= norms[:,:,None]
    imgs[ imgs < 0 ] = mask_val
    return imgs

def interpolate_run (img_gen, tags, mask, x_center, y_center, pixsize,
                    detdist, wavelen, prefix, how='fetch', interp_method='floor', 
                    radii=None, q_resolution=0, phi_resolution=0,
                    nphi=None, qmin=None, qmax=None, qmin_pix=None, qmax_pix=None,
                    detector_gain=None, index_query_fname=None):
     
    """
    Description
    ===========

    Interpolates polar rings from detector images using loki.RingData
    and saves the output to a file specified by prefix.


    Parameters
    ==========
    
    img_gen,     generator of 2d np.array float images, 
                    this should generate each image in a run
    
    tags,        list, string,  the tags associated w each image 
                generated by img_gen, (should be in order 
                with generation)
    
    mask,        2d np.array bool, a masked image, True is unmasked,
                 False is masked, should have same dimensions of 
                 images generated by img_gen
    
    x_center,    float, pixel unit where beam hits detector, 
                x dimension( fast dimension), measured from
                 0,0 pixel corner
    
    y_center,    float,  pixel unit where beam hits detector,
                 y dimension( slow dimension), measured from
                  0,0 pixel corner

    detdist,     float, sample to detector distance in meter
    
    wavelen,     float, wavelength of photons in angstroms
    
    pixsize,     float pixel size in meter
    
    prefix       str, file prefix, include directory path if necessary

    how,         str, either ('fetch' or 'polar') method of interpolating the ring. 
                
                'fetch' is the new version, which requires q_resolution 
                        and phi_resolution parameters. Uses RingData.RingFetch. 
                'polar' is the old method which makes a polar image 
                        using RingData.InterpSimple


    Required Parmeters if using 'fetch' method
    ==========================================
    interp_method,      str, should be either ['floor', 'nearest', 'nearest4', 'weighted4' ]
    
    radii,      list, range of interesting radii on the detector
                    where one wants rings
    
    q_resolution,  float , resolution of rings in inverse angstroms
    
    phi_resolution float, resolution of rings in degrees
    
    
    
    Required Parameters if using 'polar' method
    ===========================================

    nphi,        int,  azimuthal dimension of polar image, (try to keep at least 
                    single pixel resolution at qmax, you can average over polar 
                    pixels later) 
    
    qmin, qmax,         float, min q and max q bounds of each polar image 
                        being created (in inverse angstroms)
    
    qmin_pix, qmax_pix,  float, min q and max q bounds of each polar image 
                            being created (in pixel units)

    
    Optional Parameters
    ===================
    index_query_fname,    str, filename created by loki.queryRingIndices

    detector_gain,    float, absolute gain of detector
    

    Returns
    =======

    the output filename

    """

    assert( how in [ 'fetch', 'polar' ] )

    if detector_gain is None:
        detector_gain = 1
        photon_conversion_factor=1
    else:
        photon_energy = 12398.42 / wavelen
        photon_conversion_factor = detector_gain * 3.65/ photon_energy

    num_imgs = len(tags)
    output_hdf = h5py.File(  prefix + '.hdf5', 'w' )

    if how=='polar':
        
        pix2invang  = lambda qpix : np.sin(np.arctan(qpix*pixsize/detdist )/2)\
                                    *4*np.pi/wavelen 
        invang2pix  = lambda qia  : np.tan(2*np.arcsin(qia*wavelen/4/np.pi))\
                                    *detdist/pixsize
        
        assert( nphi is not None)
        
        if qmin_pix is None or qmax_pix is None:
            assert( qmin is not None)
            assert (qmax is not None)
            qmin_pix = invang2pix ( qmin )
            qmax_pix = invang2pix ( qmax )
        
#       Initialize the interpolater
        interpolater  = InterpSimple( x_center, y_center, qmax_pix, qmin_pix, nphi, 
                                        raw_img_shape=mask.shape )
#       make a polar image mask
        pmask   = interpolater.nearest( mask , dtype=bool ).round()
#       Make the polar images
        for tag in tags:
            polar_img = pmask * interpolater.nearest( img_gen.next()) \
                            * photon_conversion_factor
            output_hdf.create_dataset('ring_intensities/%s'%tag, data=polar_img)
        
        radii = np.arange( qmin_pix, qmax_pix )
        q_vals =  np.array( [ pix2invang(q_pix)
                            for q_pix in radii ] )
    
    else: #using default 'fetch' method
        assert (phi_resolution is not None)
        assert (q_resolution is not None)
        assert( radii is not None)
       
        fetcher = RingFetch( x_center, y_center, mask.shape,
                            mask, q_resolution, phi_resolution, wavelen,
                            pixsize, detdist, photon_conversion_factor, interp_method, 
                            index_query_fname)

        for tag in tags:
            fetcher.set_working_image(img_gen.next() )
            intensities = np.zeros((len(radii), fetcher.num_phi_nodes))
            for ring_index, ring_radius in enumerate( radii):
                intensities[ring_index] = \
                            fetcher.fetch_a_ring(ring_radius)
            output_hdf.create_dataset( 'ring_intensities/%s'%tag,
                                data = intensities )
#       define meta parameters not specified
        q_vals = np.array( [fetcher.r2q(q_rad) for q_rad in radii ] )
        nphi  = intensities.shape[1]
        pmask = np.ones( ( len(radii), nphi))

    phi_values = np.arange( nphi) * 2 * np.pi / nphi
#   save meta data
    output_hdf.create_dataset( 'x_center', data = x_center)
    output_hdf.create_dataset( 'y_center', data = y_center)
    output_hdf.create_dataset( 'wavelen' , data = wavelen)
    output_hdf.create_dataset( 'pixsize' , data = pixsize)
    output_hdf.create_dataset( 'detdist' , data = detdist)
    output_hdf.create_dataset( 'detgain' , data = detector_gain)
    output_hdf.create_dataset( 'photon_factor' , data = photon_conversion_factor)
    output_hdf.create_dataset( 'q_resolution' , data=q_resolution)
    output_hdf.create_dataset( 'phi_resolution', data=phi_resolution)
    output_hdf.create_dataset( 'q_mapping' , data = q_vals)
    output_hdf.create_dataset( 'num_phi', data = nphi)
    output_hdf.create_dataset( 'polar_mask', data = pmask.astype(int) ) 
    output_hdf.create_dataset( 'ring_phis', data = phi_values )
    output_hdf.create_dataset( 'q_radii', data = radii)

#   save
    print ("saving data to file %s!"%(prefix + '.hdf5'))
    output_hdf.close()
    return prefix + '.hdf5'


def make_mpccd_mask( mpccd_img, border_pad=10, mask_val=0):
    """
    Description
    ===========
    Creates a mask for the MPCCD. By default the 
    masked pixels have value `mask_val` on the
    raw image. 

    Parameters
    ==========
    `mpccd_img`    The img that the mask will be made for

    `border_pad`    Add a square border of this length
                    each masked pixel

    `mask_val`    In the raw `mpccd_img`, the masked values will be
                represented by this value (e.g. 0 or -1)

    
    Return
    ======
    `mask`    A boolean mask value with True/False for
            masked/unmasked values
    """

    mask_template = np.ones_like( mpccd_img)
    mask_template[ mpccd_img == mask_val ] = 0
    
    mask = mask_template.copy()
    for i in xrange( border_pad):
        mask = mask* np.roll(mask_template, i, axis=1)
        mask = mask* np.roll(mask_template, -i, axis=1)
        mask = mask* np.roll(mask_template, i, axis=0)
        mask = mask* np.roll(mask_template, -i, axis=0)

    return mask.astype(bool)
