# check in every run has the same basic mask
from psana import *
import numpy as np


run_nums = range(10,100)
first_mask = None
for num in run_nums:
	try:
		ds = DataSource('exp=cxilp6715:run=%d:smd'%num)
	except RuntimeError:
		continue
	det = Detector('CxiDs1.0:Cspad.0')
	det_mask = det.mask(num,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
	det_mask_image = det.image(num,det_mask).astype(bool)

	if first_mask is None:
		first_mask=det_mask_image.copy()
	else:

		if not np.all(first_mask==det_mask_image):
			print "run %d not the same as run %d"%(num, run_nums[0])
			print "mask changed at run%d"%num

			first_mask = det_mask_image.copy()


