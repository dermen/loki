import numpy as np
import h5py
import json
from loki.RingData import InterpSimple, DiffCorr

from pylab import plot,show,imshow

#####################################################
# A GROUP OF HELPER FUNCTIONS FOR VARIOUS BEAMTIMES #
#####################################################

def radProPrepJune2015(wavelen=False, pixsize=False, detdist=False, num_q=2000):
    """
    wavelen,    float , the wavelength of the run in angstroms
    pixsize,   float, the pixels size on the detector in meters
    detdist,    float, the sample to detector distance in meters
    num_q,     int, the minimum number of radial bins, dont change this 
    """
    mask    = np.load( '/home/fperakis/data/mpccd_basic_mask.npy')
    center  = np.load( '/home/fperakis/data/center.npy')
    
    if not wavelen:
        wavelen = 1.127   # ang
    if not pixsize:
        pixsize = 0.00005 # pixsize
    if not detdist:
        detdist = 0.0625  # detdist
    
    pix2invang  = lambda qpix : np.sin(np.arctan(qpix*pixsize/detdist )/2)*4*np.pi/wavelen
    qs          = [ pix2invang( q )  for q in range(num_q) ]
    return mask, center, qs 


def radProPrepFeb2014( wavelen=False, pixsize=False, detdist=False, num_q =2000 ):
    """
    wavelen,    float , the wavelength of the run in angstroms
    pixsize,   float, the pixels size on the detector in meters
    detdist,    float, the sample to detector distance in meters
    num_q,     int, the minimum number of radial bins, dont change this 
    """
    mask    = np.load( '/home/derek/data/mpccd_basic_mask.npy')
    center  = np.load( '/home/derek/data/mpccd_center_Feb2014.npy')
    
    if not wavelen:
        wavelen = 1.144   # ang
    if not pixsize:
        pixsize = 0.00005 # meter 
    if not detdist:
        detdist = 0.053  # meter
    
    pix2invang  = lambda qpix : np.sin(np.arctan(qpix*pixsize/detdist )/2)*4*np.pi/wavelen
    qs          = [ pix2invang( q )  for q in range(num_q) ]
    return mask, center, qs 


def imagesFromTags( runfile ,  tagList):
    """
    runfile,   str, filename of the run hdf5
    tagList,    list str, list of run tags
    """

    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]
    imgs_path     = '/%s/detector_2d_assembled_1'%run_key
    img_gen   = (  fh5[ imgs_path + '/' + tag + '/detector_data' ].value for tag in tagList )
    return img_gen


def getNorm( runfile, probe=True, pump=False, beam=True):
    
    fh5           = h5py.File( runfile )
    run_key       = [ k for k in fh5.keys() if k.startswith('run_') ][0]

    intens_monit = fh5['/%s/event_info/bl_3/oh_2/bm_2_pulse_energy_in_joule'%run_key].value

    beam_stat     = fh5['/%s/event_info/acc/accelerator_status'%run_key].value.astype(bool)
    pump_stat     = fh5['/%s/event_info/bl_3/lh_1/laser_pulse_selector_status'%run_key].value.astype(bool)
    probe_stat    = fh5['/%s/event_info/bl_3/eh_1/xfel_pulse_selector_status'%run_key].value.astype(bool)
    
    intens_monit  = [ val for i,val in enumerate( intens_monit) if beam_stat[i] == beam and pump_stat[i] == pump and probe_stat[i] == probe ]

    return np.sum( intens_monit)

def selectImagesSimple(  runfile, shutter=1, beam=1 ):
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
    shutt_stat    = fh5['/%s/event_info/bl_3/eh_1/xfel_pulse_selector_status'%run_key].value

#   keep only the tags corresponding to status
    tags     = [ tag for i,tag in enumerate( tags) if beam_stat[i] == beam and shutt_stat[i] == shutter ]
    img_gen  = (  fh5['%s/detector_2d_assembled_1/%s/detector_data'%(run_key,tag) ].value for tag in tags )
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
    beam_stat     = fh5['/%s/event_info/acc/accelerator_status'%run_key].value.astype(bool)
    pump_stat     = fh5['/%s/event_info/bl_3/lh_1/laser_pulse_selector_status'%run_key].value.astype(bool)
    probe_stat    = fh5['/%s/event_info/bl_3/eh_1/xfel_pulse_selector_status'%run_key].value.astype(bool)

#   keep only the tags corresponding to status
    tags         = [ tag for i,tag in enumerate( tags) if beam_stat[i] == beam and pump_stat[i] == pump and probe_stat[i] == probe ]
    img_gen   = (  fh5['%s/detector_2d_assembled_1/%s/detector_data'%(run_key,tag) ].value for tag in tags )
    num_im = len(tags)
    return img_gen, num_im

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

def interpolate_run ( img_gen, tags, mask, x_center, y_center, qmin, qmax, pixsize, detdist, wavelen, nphi, prefix):
    """
    run_number,  string rung number
    img_gen,     generator of 2d np.array float images, this should generate each image in a run
    tags,        list, string,  the tags associated w each image generated by img_gen, (should be in order with generation)
    mask,        2d np.array bool, a masked image, True is unmasked, False is masked, should have same dimensions of images generated by img_gen
    x_center,    float, pixel unit where beam hits detector, x dimension( fast dimension), measured from 0,0 pixel corner
    y_center,    float,  pixel unit where beam hits detector, y dimension( slow dimension), measured from 0,0 pixel corner
    qmin, qmax,  float, min q and max q bounds of each polar image being created (in inverse angstroms)
    pixsize,     float pixel size in meter
    detdist,     float, sample to detector distance in meter
    wavelen,     float, wavelength of photons in angstroms
    nphi,        int,  azimuthal dimension of polar image, (try to keep at least single pixel resolution at qmax, you can average over polar pixels later) 
    prefix       str, file prefix, this script makes two files, (this parameter will be the prefix of each)
    """

    num_imgs = len(tags)

    output_hdf = h5py.File(  prefix + '.hdf5', 'w' )

#   some useful functions
    pix2invang  = lambda qpix : np.sin(np.arctan(qpix*pixsize/detdist )/2)*4*np.pi/wavelen 
    invang2pix  = lambda qia  : np.tan(2*np.arcsin(qia*wavelen/4/np.pi))*detdist/pixsize

    qmin_pix = invang2pix ( qmin )
    qmax_pix = invang2pix ( qmax )

#   Initialize the interpolater
    interpolater  = InterpSimple( x_center, y_center, qmax_pix, qmin_pix, nphi, raw_img_shape = mask.shape )

#   make a polar image mask
    pmask   = interpolater.nearest( mask , dtype=bool ).round()

#   Make the polar images
    polar_imgs = np.array( [ pmask * interpolater.nearest( img_gen.next() ) for dummie in range( num_imgs) ] )

#   save the data raw
    output_hdf.create_dataset( 'polar_data',data = polar_imgs)

#   Consider normalizing the images 
    polar_imgs = normalize_polar_images( polar_imgs) 

#   save the data normalized
    output_hdf.create_dataset( 'normalized_polar_data',data = polar_imgs)

#   save a lookup-map
    tag_map = {}
    for indx,tag in enumerate( tags ):
        tag_map[tag] = indx

#   save meta data
    output_hdf.create_dataset( 'polar_mask',data = pmask.astype(int))
    output_hdf.create_dataset( 'x_center',   data = x_center)
    output_hdf.create_dataset( 'y_center',   data = y_center)
    output_hdf.create_dataset( 'num_phi',   data = nphi)
    output_hdf.create_dataset( 'wavelen' ,   data = wavelen)
    output_hdf.create_dataset( 'pixsize' ,     data = pixsize)
    output_hdf.create_dataset( 'detdist' ,     data = detdist)

#   save the q-mapping
    qrange_pix = np.arange( qmin_pix, qmax_pix )
    q_map       =  np.array( [ [ ind, pix2invang(q) ] for ind,q in enumerate( qrange_pix) ] )
    output_hdf.create_dataset( 'q_mapping' ,    data = q_map )

#   save
    print ("saving data to file %s!"%(prefix + '.hdf5'))
    output_hdf.close()

#   save the lookup-map
    print( "saving data dictionary to file %s!"%(prefix + '.json') )
    dump_file = open( prefix + '.json', 'w')
    json.dump( tag_map, dump_file )
    dump_file.close()

    return prefix + '.hdf5', prefix + '.json' # return the file names it created


def correlate_polar_images( data_hdf_fname, tag_map_fname, outfile, tag_pairs_fname=None ):
    """
    data_hdf_fname,     string, hdf5 file name, hdf data file returned by interpolate_run 
    tag_map_fname,      sting, JSON file name,  tag map file returned by interpolate_run
    outfile,            string, where to store the output
    tag_pairs_fname,    string, JSON file name, contains a list of exposure tag pairs,
                                                and each pair will be loaded, and subtracted
    """

#   load the polar data
    data_hdf = h5py.File( data_hdf_fname )

    polar_data = data_hdf['normalized_polar_data'][:10] #.value
    polar_mask = data_hdf['polar_mask'].value
    q_map  = { i: val for i,val in data_hdf[ 'q_mapping'].value }
    nphi = data_hdf['num_phi'].value
    nq = len( q_map)

###############################
#                             #
#   PROCESS POLAR DATA HERE   #
#                             # 
    #norm = np.ma.masked_equal( polar_data*polar_mask,0).mean(1).mean(1)
    #norm = norm[:,None,None] #expand dimensions
    #polar_data = polar_data / norm
###############################

#   load out map
    tag_map = json.load( open( tag_map_fname) )

    if tag_pairs_fname is not None:
#       load the exposure tag pairs
        tag_pairs = json.load( open( tag_pairs_fname ))
        exposure_diffs = [] 
        for i_, tags in enumerate( tag_pairs):
            tagA, tagB = tags
            try: indxA = tag_map[ tagA]
            except KeyError: continue
            try: indxB = tag_map[ tagB]
            except KeyError: continue
            shotA = polar_data[ indxA ]
            shotB  = polar_data[ indxB]
            exposure_diffs.append( shotA - shotB )
        exposure_diffs = np.vstack( exosure_diffs ) 
        DC = DiffCorr( exposure_diffs ) 
    else:
        DC = DiffCorr( polar_data, pre_dif=False, delta_shot=2)
#   take the autocorrelation of each pair
    cor = DC.autocorr(num_high=10)
#################################
#                               #
#   PROCESS CORRELATIONS HERE   #
#                               #
#################################
#   take the mean over pairs
    cor_m = cor.mean(0)

    output_hdf = h5py.File( outfile, 'w')
    #output_hdf.create_dataset( 'all_cor', data=cor)
    output_hdf.create_dataset( 'ave_cor', data=cor_m)
    if tag_pairs_fname is not None:
        output_hdf.create_dataset( 'tag_pairs', data=tag_pairs)
    output_hdf.close()

