from psana import *


import sys
import os
import argparse
import numpy as np

from loki.RingData import InterpSimple


from PSCalib.GeometryAccess import GeometryAccess

######
# parse parameters
#######


parser = argparse.ArgumentParser(description='Analyze a run. Use MPI!')
parser.add_argument('-r', '--run', type=int, 
    required=True, help='run number to process')
parser.add_argument('-m', '--max', type=int, 
    default=None, help='max shots to process')
parser.add_argument('-c', '--chunk', 
    default=250, type=int, help='shots per sub-file')
parser.add_argument('-s','--saveDir',type=str,
    required=True, help='directory in which to save the interpolated data')


parser.add_argument('-d','--det_dist',type = float,
    required=True, help='detector distance in micron (um)')
parser.add_argument('-t','--center',type=str,
    required=True, help='path to .npy file that contains the\
     center of the detector in pixel units')

parser.add_argument('-qi' ,'--qmin', type = float,
    required=True, help='min q value approximate) to interpolate in inverse angstrom')
parser.add_argument('-qf','--qmax',type = float,
    required=True,help='max q value approximate) to interpolate in inverse angstrom')

parser.add_argument('-b', '--bin_fac', type=float,
    default=None,help='factor by which to bin pixels in detector image')

parser.add_argument('-p','--nphi',type=int,
    default = 360, help='number of phi values in the interpolated data')

parser.add_argument('-n', '--num_nodes',
required=True, type=int, help='number of nodes used in mpirun. \n NEEDS TO MATCH mpirun -n NUM_NODES!!!')


args=parser.parse_args()

######
# experimental parameters
#######

pix_size = 110. # um
det_dist = args.det_dist # 80000 um
ring_center = np.load(args.center) #(1731/2, 1738/2) # center of q rings coordinates in pixels 
#if binning pixels while interpolating, then this need to be centers from binned images


wavlen = 1.24 #angstrom
k_beam = np.pi*2/wavlen # inverse angstrom

bright_threshold = 2.**14 * 0.95 # if pixels it with 95% of the maximum read out, mask it 

#######
# interpolation parameters
#######
# range of q values
qmin= args.qmin # 0.2 inverse angstrom
qmax = args.qmax # 0.8
theta_min = np.arcsin (qmin/ (2. * k_beam))
theta_max = np.arcsin (qmax/ (2. * k_beam))

bin_fac = args.bin_fac
# compute qmin and qmax in pixel unit
if bin_fac is None:
    qRmin = int( np.tan(2*theta_min) * det_dist / pix_size )
    qRmax = int( np.tan(2*theta_max) * det_dist / pix_size )

else:
    # adjust ring_center is bin_fac is not None
    ring_center = (ring_center[0]/bin_fac, ring_center[1]/bin_fac)
    
    qRmin = int( np.tan(2*theta_min) * det_dist / pix_size / bin_fac)
    qRmax = int( np.tan(2*theta_max) * det_dist / pix_size / bin_fac)

q_inds = np.array( [qRmin, qRmax] )

num_phi = args.nphi # default 360

#######
# data saving parameters
#######
# full path to small data, including run directory
smldir = os.path.join(args.saveDir, 'run%04d'%args.run)
if not os.path.exists(smldir):
    os.makedirs(smldir)
smlpath = os.path.join( smldir, 'run%04d-%d.h5')

# check chunk size
run = args.run #117
if args.max is None:
    max_event = sys.maxint
else:
    max_event = int(args.max/args.num_nodes)
chunk=  int(args.chunk/args.num_nodes) # numbrt of shots per chunk

# string to data source
ds_string = 'exp=xpptut15:run=%s:smd' % str(run)
print 'Processing: %s' % ds_string
ds = MPIDataSource(ds_string)

#declare detector
cspaddet = Detector('cspad')

#initilize file counting and shot counting
file_counter = 1
shot_counter = 0
total_shots = 0
smlfname = smlpath % (run, file_counter)
smldata = ds.small_data(smlfname)


image_sum = None

for nevt,evt in enumerate(ds.events()):
    # check if max event is reached
    if nevt > (max_event-1): break

    # check if chunk is finished and should save
    if shot_counter > 0 and (not shot_counter % chunk):
        print "\tSaving data to %s."%smlfname
        overall_img_sum = smldata.sum(image_sum/(shot_counter*args.num_nodes) )
        print('summing all shots and total shots is:%d'% (shot_counter*args.num_nodes) )
        

        # get "summary" data
        params = {'cspad': {'qPixels':q_inds, 'bin_fac': bin_fac} }
        smldata.save(params)
        smldata.save(average_shot = overall_img_sum)
        smldata.save()

        #close file
        smldata.close()
        # reset shot counter and add to file counter
        shot_counter = 0
        file_counter += 1
        # reset image sum
        image_sum = None
        # new small data object

        smlfname = smlpath % (run, file_counter)
        smldata = ds.small_data(smlfname)

    # get img
    img = cspaddet.image(evt)
    if img is None: continue

    # sum all the images in this run
    if image_sum is None:
        image_sum = img
    else:
        image_sum += img

    # interpolation begins here
    if shot_counter == 0:
      # create SimpleInterp object if it's the first shot
        img_shape = img.shape

        interpolator = InterpSimple (ring_center[0], ring_center[1],
         qRmax, qRmin,
         num_phi, img_shape,
         bin_fac = bin_fac,
         use_zoom = False)
    # threshold mask
    threshold_mask = img<bright_threshold

    # apply threshold to img directly
    img *= threshold_mask
    polar_interp = interpolator.nearest_naive_bin( img )

    d = {'cspad':{'polar_intensity': polar_interp}}

    # save per-event data
    smldata.event(d)
    shot_counter += 1
    total_shots +=1


# if there are more shots to save in a separate file
if shot_counter>0:
    remainder_shots = smldata.sum(shot_counter)
    if remainder_shots is not None:
        print("number of shots in the last file is: %d"%remainder_shots)
    
    print "\tSaving data to %s."%smlfname
    overall_img_sum = smldata.sum(image_sum/1000. )# the factor of 1000 is there to 
    # prevent overflow
    
    if remainder_shots is not None:
        overall_img_sum *= (float(1000.)/remainder_shots)

    #if smldata.master:
    smldata.save(average_shot = overall_img_sum)

    params = {'cspad': {'qPixels':q_inds, 'bin_fac': bin_fac} }
    smldata.save(params)
    smldata.close()
print 'SMALLDATA DONE'

