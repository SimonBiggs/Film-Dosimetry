from matplotlib.pyplot import *
from matplotlib.image import *

from numpy import *
from scipy.optimize import curve_fit

from glob import glob
import os

from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed, RotatingMarker

from film_functions import fitting_func, dose2density, save_dict


# Defines the imageset
imageset = '004_Matt_brachy_aged_2'

# Pulls the calibration file paths names
calibrationDirectories = glob('../image_sets/'+imageset+'/calibration/*')
numDirectories = shape(calibrationDirectories)[0]

# Initialises the dose and channel varaibles
cal_D_fitting = array([])
cal_OD_fitting = array([[],[],[]])

cal_D = {}

cal_ODr = {}
cal_ODg = {}
cal_ODb = {}


widgets = ['Loading images: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets = widgets, maxval=numDirectories).start()


for i in range(numDirectories):
	
	# Images in current directory
	currentImages = glob(calibrationDirectories[i]+'/*.png')
	numImages = shape(currentImages)[0]
	
	cal_D[i] = float(os.path.basename(calibrationDirectories[i]))
	
	cal_ODr[i] = array([])
	cal_ODg[i] = array([])
	cal_ODb[i] = array([])
	
	for j in range(numImages):
		im = imread(currentImages[j])
		
		cal_ODr[i] = append (
			cal_ODr[i],
			-log( ravel(im[:,:,0]) )
		)
		
		cal_ODg[i] = append (
			cal_ODg[i],
			-log( ravel(im[:,:,1]) )
		)
		
		cal_ODb[i] = append (
			cal_ODb[i],
			-log( ravel(im[:,:,2]) )
		)
	
	# Formats data for fitting
	cal_D_fitting = append(cal_D_fitting,cal_D[i] * ones(shape(cal_ODr[i])),axis=1)
	cal_OD_fitting = append(cal_OD_fitting,[cal_ODr[i],cal_ODg[i],cal_ODb[i]],axis=1)
	
	pbar.update(i+1)
	
pbar.finish()
	
fit = {}


widgets = ['Curve fitting: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets = widgets, maxval=numDirectories).start()

pbar = ProgressBar(widgets = widgets, maxval=3).start()
# Using least squares fitting calculates the parameters for each of the colour channels
fit['red'], redLsqCov = curve_fit(fitting_func,cal_D_fitting,cal_OD_fitting[0,:])
fit['redSig'] = sqrt(diag(redLsqCov))
pbar.update(1)

fit['green'], greenLsqCov = curve_fit(fitting_func,cal_D_fitting,cal_OD_fitting[1,:])
fit['greenSig'] = sqrt(diag(greenLsqCov))
pbar.update(2)

fit['blue'], blueLsqCov = curve_fit(fitting_func,cal_D_fitting,cal_OD_fitting[2,:])
fit['blueSig'] = sqrt(diag(blueLsqCov))
pbar.update(3)

pbar.finish()



save_dict(fit,'../image_sets/'+imageset+'/data/calibration')


figure(1)
clf()

# Defines the xi values for plotting fits
dosei = linspace(cal_D_fitting.min(),cal_D_fitting.max(),100)

plot(cal_D_fitting,cal_OD_fitting[0,:], 'r.')
plot(dosei,dose2density(dosei, fit['red']), 'r-')

plot(cal_D_fitting,cal_OD_fitting[1,:], 'g.')
plot(dosei,dose2density(dosei, fit['green']), 'g-')

plot(cal_D_fitting,cal_OD_fitting[2,:], 'b.')
plot(dosei,dose2density(dosei, fit['blue']), 'b-')

show()