import numpy as np
import pylab as plt
# uncomment out if you want pretty plots
plt.style.use('ggplot') 
blue = '#348ABD'
red = '#E24A33'

import RingData
#import


# load the testing image!
img = np.load('test_img.npy')

# guess where the forward beam hits the detector
# last dimension is fast (horizontal for 2D) dimension in numpy arrays ;)
center_guess = ( 2+img.shape[1] / 2., 1+img.shape[0] / 2. )

# make a mask that has the same shape as img (True is unmasked, False is masked)
mask = np.load( 'test_mask.npy' )

# make a radial profile
# and then we can pick a peak for measuring the center..
# Usually there is a calibration sample  e.g. gold nanoparticle diffraction
# that exhibits bright rings which can be used to optimize center..

# this is an approximate radial profile because the center is inaccurate
RP = RingData.RadialProfile( center_guess, 
                        img_shape=img.shape, mask=mask, minlength=1900  ) 
radial_profile = RP.calculate(img )

###################
# PLOT
fig1 = plt.figure(1)
ax = plt.gca()
ax.tick_params(which='both', labelsize=12, length=0)
ax.plot( radial_profile , marker='s', color='Darkorange') 
ax.set_xlabel( 'Radial pixel unit', fontsize=12 )# , labelpad=15)
ax.set_ylabel('CCD count', fontsize=12, labelpad=15)
ax.set_xlim(10,130)
ax.grid(lw=1, alpha=0.5, color='#777777', ls='--' )
ax.set_axis_bgcolor('w')


# from the plot choose a nice looking peak
peak_radius_guess = 16
ring_param_guess = ( center_guess[0], center_guess[1], peak_radius_guess )

# use the guessed paraemters to optimize the center coordinate
RF = RingData.RingFit(img)
x_center,y_center,peak_radius = RF.fit_circle( ring_param_guess, 
                                num_fitting_pts=20, 
                                ring_width=10, num_high_pix=2 )

fig2 = plt.figure(2)
plt.subplot(121)
ax = plt.gca()
ax.set_xticks( [] )
ax.set_yticks( [] )
ax.imshow(img, cmap='hot', interpolation='nearest')
circ = plt.Circle( xy=(x_center, y_center), radius=peak_radius, 
                ec='c', lw=2, fc='none', ls='dashed' ) 
ax.add_patch( circ)

plt.subplot(122)
ax = plt.gca()
ax.set_xticks( [] )
ax.set_yticks( [] )
# zoom in (thats what the 125/ 175 business is doing, nevermind it
ax.imshow(img[125:175,125:175], cmap='hot', interpolation='nearest')
circ = plt.Circle( xy=(x_center-125, y_center-125), radius=peak_radius, 
                ec='c', lw=2, fc='none', ls='dashed' ) 
ax.add_patch(circ)


# now make another radial profile with the correct center
# might not change much but we are trying to be
# precise here!

RP.update_center( (x_center, y_center) )
refined_radial_profile = RP.calculate(img )

fig3 = plt.figure(3)
ax = plt.gca()
ax.tick_params(which='both', labelsize=12, length=0)
ax.plot( radial_profile , ms=7, marker='s', 
            color='Darkorange', label='old center') 
ax.plot( refined_radial_profile , ms=7, marker='o', color=blue, 
                label='new center', alpha=.9) 
ax.set_xlabel( 'Radial pixel unit', fontsize=12 )#, labelpad=15)
ax.set_ylabel('CCD count', fontsize=12, labelpad=15)
ax.set_xlim(10,130)
ax.grid(lw=1, alpha=0.5, color='#777777', ls='--' )
leg = ax.legend()
fr = leg.get_frame()
fr.set_facecolor('w')
fr.set_alpha(.7)
ax.set_axis_bgcolor('w')
ax.set_ylim(4500,9500)
ax.set_xlim(10,70)

# set a radius-of-interest from the radial profile
interesting_peak_radius = 59

##########################
# EXPERIMENTAL PARAMETERS
#########################
# sacla pixel size
pixsize = 0.00005 # meters

# test image resolution reduction
pixsize = pixsize * 8

# sample to detector distance
detdist = 0.051 # meters

# x-ray photon wavelength
wavelen = 1.41 # angstroms

# Make the CCD -> photon conversion factor
# detector absolute gain
gain = 16.912
# photon energy
energy = 12398.42 / wavelen  # electron-Volts
# factor
photon_conversion_factor = gain * 3.65 / energy

RingFetch = RingData.RingFetch( x_center, y_center, img, mask=mask, 
                        q_resolution=0.05, phi_resolution=1.25, 
                        wavelen=wavelen, pixsize=pixsize, detdist=detdist,
                        photon_conversion_factor=photon_conversion_factor)
phis, ring = RingFetch.fetch_a_ring( interesting_peak_radius)


fig4 = plt.figure(4)
plt.plot( phis, ring * photon_conversion_factor, lw=3, color=blue )
ax = plt.gca()
ax.tick_params(which='both', labelsize=12, length=0)
ax.set_xlabel(r'$\phi\,\,(0-2\pi)$', fontsize=12, labelpad=10 )
ax.set_ylabel(r'Photon counts', fontsize=12, labelpad=15)
ax.grid(lw=1, alpha=0.5, color='#777777', ls='--' )
ax.set_axis_bgcolor('w')
ax.set_xlim(0,2*np.pi)



plt.show()
