import numpy as np
import h5py


from loki.RingData import InterpSimple, RingFetch, RadialProfile
from loki.utils.postproc_helper import smooth

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

def get_detector_gain(runFile):
    f = h5py.File( runFile, 'r')
    detgain = f[f.keys()[1]+\
        '/detector_2d_assembled_1/detector_info' +'/absolute_gain']\
        .value
    return detgain

def get_pixel_size(runFile):
    f = h5py.File( runFile, 'r')
    pixsize = f[f.keys()[1]+\
        '/detector_2d_assembled_1/detector_info' \
        +'/pixel_size_in_micro_meter'].value[0] / 1e6 
    return pixsize

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
                        beamline=3, hutch=1, return_extra=False,
                        energy=False):
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
   
    energy_stat = fh5['/%s/event_info/bl_3/oh_2/photon_energy_in_eV'%run_key].value

    shutt_params = (run_key, beamline, hutch)
    shutt_stat_path = '/%s/event_info/bl_%d/eh_%d/xfel_pulse_selector_status'\
                        %shutt_params
    shutt_stat    = fh5[shutt_stat_path].value

#   keep only the tags corresponding to status
    tags, energies     = zip( *[ (tag,energy_stat[i]) for i,tag in enumerate( tags) 
            if beam_stat[i] == beam and shutt_stat[i] == shutter ])

    img_path = '%s/detector_2d_assembled_1/%s/detector_data'
    img_gen  = (  fh5[img_path%(run_key,tag) ].value 
            for tag in tags )
    
    if return_extra:
        run = run_key.split('_')[1], 
        return img_gen, tags, run
    else:
        if energy:
            return img_gen, tags, energy_stat
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
                     ring_locations=None, q_resolution=0, phi_resolution=0, radius_unit='inv_ang',
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
    
    ring_locations,      list, range of ring radii or ring momentum transfer magnitudes 
    
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

    assert( radius_unit in [ 'inv_ang', 'pixels'])

    num_imgs = len(tags)

    if isinstance(wavelen,(int float, long, complex)) and isinstance(detdist,(int float, long, complex)):
       
        wavelen, detdist = [wavelen]*num_imgs, [detdist]*num_imgs
       
    else:

        assert( isinstance( wavelen, (list, np.ndarray)))
        assert( isinstance( detdist, (list, np.ndarray)))
        assert( len(wavelen) == len(detdist) ==  num_imgs )
    
    if detector_gain is None:
        
        detector_gain, photon_conversion_factor = -1, [1]*num_imgs
        
    else:
        
        photon_conversion_factor = [detector_gain * 3.65 * w / 12398.42 for w in wavelen]

    with h5py.File(  prefix + '.hdf5', 'w' ) as output_hdf:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if how=='polar':
           

            assert( nphi is not None)
            

            for i_tag, tag in enumerate(tags):
                
                pix2invang  = lambda qpix : np.sin(np.arctan(qpix*pixsize/detdist[i_tag] )/2)\
                                            *4*np.pi/wavelen[i_tag]
                
                invang2pix  = lambda qia  : np.tan(2*np.arcsin(qia*wavelen[i_tag]/4/np.pi))\
                                            *detdist[i_tag]/pixsize
                
                if qmin_pix is None or qmax_pix is None:
                
                    assert( qmin is not None)
                    
                    assert (qmax is not None)
                    
                    qmin_pix = invang2pix (qmin)
                    
                    qmax_pix = invang2pix (qmax)
                
#           Initialize the interpolater
                interpolater  = InterpSimple( x_center, y_center, qmax_pix, qmin_pix, nphi, 
                                                raw_img_shape=mask.shape )
#           make a polar image mask
                pmask   = interpolater.nearest( mask , dtype=bool ).round()

#           Make the polar images
                polar_img = pmask * interpolater.nearest( img_gen.next()) \
                                * photon_conversion_factor[i_tag]

                output_hdf.create_dataset('ring_intensities/%s'%tag, data=polar_img, dtype=np.float32)
                
                output_hdf.create_dataset('ring_mask/%s'%tag, data=pmask.astype(np.int8), dtype=np.int8)
                
                ring_radii =  np.arange(qmin_pix, qmax_pix) 
                
                ring_mag =  np.array( [pix2invang(q_pix)for q_pix in ring_radii])
                
                output_hdf.create_dataset( 'ring_radii/%s'%tag, data = ring_radii)
                
                output_hdf.create_dataset( 'ring_momentum_transfer/%s'%tag, data = ring_mag)
               
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else: # using default 'fetch' method
            assert(phi_resolution is not None)
            
            assert(q_resolution is not None)
            
            assert(ring_locations is not None)
           
            fetcher = RingFetch( 
                        a=x_center, 
                        b=y_center, 
                        img_shape=mask.shape,
                        mask=mask, 
                        q_resolution=q_resolution, 
                        phi_resolution=phi_resolution,
                        pixsize=pixsize, 
                        interp_method=interp_method, 
                        index_query_fname=index_query_fname)

            ring_radii = np.zeros( (num_imgs, len(ring_locations) ))
            ring_mag = np.zeros_like( ring_radii)
            
            for i_tag, tag in enumerate(tags):

                fetcher.set_params(wavelen[i_tag], detdist[i_tag])
                
                fetcher.set_photon_factor(photon_conversion_factor[i_tag])
                
                fetcher.set_working_image(img_gen.next() )

                intensities = np.zeros((len(ring_locations), fetcher.num_phi_nodes))
                
                if radius_unit=='inv_ang':
                    
                    for ring_index, ring_q in enumerate( ring_locations):
                        
                        intensities[ring_index] = \
                                    fetcher.fetch_a_ring(q=ring_q)
                    
                        ring_radii[ i_tag, ring_index  ] = int(round(fetcher.q2r(ring_q)))
                        
                        ring_mag[ i_tag, ring_index] = ring_q
                
                else:
                        
                    for ring_index, ring_radius in enumerate( ring_locations):
                    
                        intensities[ring_index] = \
                                    fetcher.fetch_a_ring(radius=ring_radius)
                        
                        ring_radii[ i_tag, ring_index  ] = ring_radius
                        
                        ring_mag[ i_tag, ring_index] = fetcher.r2q(ring_radius)

                output_hdf.create_dataset( 'ring_intensities/%s'%tag,
                                    data = intensities, dtype=np.float32 )

#       define meta parameters not specified
            nphi  = intensities.shape[1]
            
            pmask = np.ones((len(ring_locations), nphi))
        
            output_hdf.create_dataset( 'ring_radii', data = ring_radii, dtype=np.float32)
            
            output_hdf.create_dataset( 'ring_moementum_transfer', data=ring_mag, dtype=np.float32)
            
            output_hdf.create_dataset( 'ring_mask', data = pmask.astype(np.int8)) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        phi_values = np.arange( nphi) * 2 * np.pi / nphi

#   save meta data
        output_hdf.create_dataset('how', data=how)
        output_hdf.create_dataset('interp_method', data=interp_method)
        
        output_hdf.create_dataset( 'x_center', data = x_center, dtype=np.float32)
        output_hdf.create_dataset( 'y_center', data = y_center, dtype=np.float32)
        output_hdf.create_dataset( 'wavelen' , data = wavelen, dtype=np.float32)
        output_hdf.create_dataset( 'pixsize' , data = pixsize, dtype=np.float32)
        output_hdf.create_dataset( 'detdist' , data = detdist, dtype=np.float32)
        output_hdf.create_dataset( 'detgain' , data = detector_gain, dtype=np.float32)
        output_hdf.create_dataset( 'photon_factor' , data = photon_conversion_factor, dtype=np.float32)
        output_hdf.create_dataset( 'q_resolution' , data=q_resolution ,dtype=np.float32)
        output_hdf.create_dataset( 'phi_resolution', data=phi_resolution, dtype=np.float32)
        output_hdf.create_dataset( 'num_phi', data = nphi, dtype=np.float32)
        output_hdf.create_dataset( 'ring_phis', data=phi_values, dtype=np.float32 )

#   save
        print ("saving data to file %s!"%(prefix + '.hdf5'))
    #output_hdf.close()
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



def calibrate_parameters( rad1, rad2, q1_sim, q2_sim, detdist_scan, wavelen_scan, 
                        pixsize=0.00005): 
    """ 
    Using two Bragg rings, calibrate the photon wavelength and sample-to-detector position
    for the given experiment..

    `rad1` `rad2`       radius of two Bragg rings in pixel units

    `q1_sim` `q2_sim`   Corresponding theortical moementum transfer magnitude of 
                        the Bragg rings in inverse angstroms
    
    `detdist_scan`      range of detector distances to scan in meters
    
    `wavelen_scan`      range of wavelengths to scan in angstroms
    
    `pixsize`           size of the pixel in meters
    """
    

    def diff_q(DETDIST, WAVELEN):

        q1 = np.sin(np.arctan( rad1*pixsize/DETDIST )/2.)*4.*np.pi/ WAVELEN
    
        q2 = np.sin(np.arctan( rad2*pixsize/DETDIST )/2.)*4.*np.pi/ WAVELEN
        
        return np.abs( q1_sim - q1) + np.abs( q2_sim-q2)

    
    param_pairs = np.array( [ (detdist, wavelen, diff_q(detdist,wavelen) ) 
                                
                                for detdist in detdist_scan 
                                    
                                    for wavelen in wavelen_scan ] )

    min_ind = np.argmin( param_pairs[:,2] )

    detdist_correct, wavelen_correct, callibration_score =  param_pairs[min_ind]
    

    return detdist_correct, wavelen_correct, callibration_score


def calibrate( img_gen, center, wavescan, detscan, q1_sim=2.48, q2_sim=2.99,
                wavelen_guess=1.41, detdist_guess=0.051, speed='slow',
                mask=None, index_query_fname=None, scan_width=50,
                beta=20, window_size=40, factor=0.007, pixsize=0.00005):

    """Use this function to calibrate the detector distance and the wavelength 
    for a particular group of images (e.g. from a run)"""

    rad1_guess = int(round(\
                    np.tan( 2*np.arcsin(q1_sim*wavelen_guess/4./np.pi) )*detdist_guess/pixsize\
                    ))
    rad2_guess = int(round(\
                    np.tan( 2*np.arcsin(q2_sim*wavelen_guess/4./np.pi) )*detdist_guess/pixsize\
                    ))
    rad1_scan = np.arange( rad1_guess-scan_width, rad1_guess+scan_width)
    rad2_scan = np.arange( rad2_guess-scan_width, rad2_guess+scan_width)

    radpro = RadialProfile( center, img_shape=mask.shape, mask=mask)
    radpro.set_params(
            wavelen=wavelen_guess,
            detdist=detdist_guess,
            pixsize=pixsize,
            factor=factor)

    detdists, wavelens, scores = [], [], []

    for img in img_gen:
        print ("\nCalibrating a new image...")
        if speed=='slow':
            rp1 = radpro.calculate_using_fetch( img, rad1_scan, index_query_fname=index_query_fname)
            rp2 = radpro.calculate_using_fetch( img, rad2_scan, index_query_fname=index_query_fname)
        else:
            rp1 = radpro.calculate(img)[rad1_scan]
            rp2 = radpro.calculate(img)[rad2_scan]
        
        sm_rp1 = smooth(rp1, beta, window_size)
        sm_rp2 = smooth(rp2, beta, window_size)
        r1 = np.argmax( sm_rp1 ) + rad1_scan[0]
        r2 = np.argmax( sm_rp2 ) + rad2_scan[0]

        print( "  Found ring1 at %d compared to %d."%(r1,rad1_guess))
        print( "  Found ring2 at %d compared to %d."%(r2,rad2_guess))
        print("  Calibrating...")

        detdist, wavelen, score = calibrate_parameters(
                                r1,
                                r2,
                                q1_sim,
                                q2_sim,
                                detscan,
                                wavescan,
                                pixsize)

        print("  New detector distance calibrated from %.4f --> %.4f"%(detdist_guess, detdist))
        print("  New wavelength calibrated from %.4f --> %.4f"%(wavelen_guess, wavelen))

        rad1 = int(round(\
                        np.tan( 2*np.arcsin(q1_sim*wavelen/4./np.pi) )*detdist/pixsize\
                        ))
        rad2 = int(round(\
                        np.tan( 2*np.arcsin(q2_sim*wavelen/4./np.pi) )*detdist/pixsize\
                        ))

        detdists.append(detdist)
        wavelens.append(wavelen)
        scores.append(score)

    return detdists, wavelens, scores

