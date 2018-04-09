from psana import *
import numpy as np
import h5py
import sys
import scipy

run_begin = int(sys.argv[1])
run_end = int(sys.argv[2])
runnums = range(run_begin,run_end)

def get_imgs(evets):
    imgs=[]
    for nevent,evt in enumerate(events):
        if nevent>=1000: 
            break
        try:
            img = det.image(evt)
        except ValueError:
            return None
        imgs.append(img)
    imgs=np.array(imgs)

    return imgs

for run in runnums:

    # make mask for this run
    ds_str = 'exp=cxilp6715:run=%d:smd' % run
    try:
        ds = MPIDataSource(ds_str)
    except RuntimeError:
        print("run %d does not exit"%run)
        continue
    print('making masks for run %d'%run)
    det = Detector('CxiDs1.0:Cspad.0')
    det_mask = det.mask(run,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
    det_mask = det.image(run,det_mask).astype(bool)

    events = ds.events()
    # now get the negative pixel mask from gathering a few hundred images
 
    imgs=get_imgs(events)
    if imgs is None:
        print("run %d does not exit"%run)
        continue
    im = np.median(imgs,0)
    negative_pixel_mask=im>0
    negative_pixel_mask = ~(scipy.ndimage.morphology.binary_dilation(~negative_pixel_mask, iterations=1) )

    with h5py.File('/reg/d/psdm/cxi/cxilp6715/results/masks_files/run%d_masks.h5'%run,'w') as f_out:
        f_out.create_dataset('negative_pixel_mask',data=negative_pixel_mask)
        f_out.create_dataset('psana_mask', data=det_mask)
        f_out.create_dataset('mask',data=det_mask*negative_pixel_mask)
print('Done!')  