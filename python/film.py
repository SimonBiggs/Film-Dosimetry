# Defines the file to look in for the images
imageset = '001_Matt_brachy'

# File type of the images
filetype = 'png'

# The pixel to cm conversion. The images in question use 70dpi
pixel2dist = 2.54/70

# Defines margin to be removed on the film measurement
xmargin = 2
ymargin = 0.5


# from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *
from matplotlib.image import *
from numpy import *
# from scipy.interpolate import *
from scipy.optimize import *
from glob import glob
import os


def fitting_func(x, a, b, c):
	# The form for the fitting -- From Micke (2011)
  return -log((a + b*x)/(c + x))

def dose2PixelVal(dose, param):
	# Easier use with the output of curve_fit
	pixelVal = fitting_func(dose, param[0], param[1], param[2])
	return pixelVal
	
def pixelVal2Dose(pixelVal, param):
	dose = (param[2] * exp(-pixelVal) - param[0]) / (param[1] - exp(-pixelVal))
	return dose

def pull_filename(fullPath):
	# Converts full path name into just the file name with extension removed
	# Used to make titles and to pull Dose value out of file names
	filename = os.path.basename(fullPath)
	result = filename[0:len(filename)-4]
	return result



# Pulls the calibration file paths names
calibrationFiles = glob('../image_sets/'+imageset+'/calibration/*.'+filetype)
numFiles = shape(calibrationFiles)[0]

# Pulls the measurement file paths names
measurementFiles = glob('../image_sets/'+imageset+'/measurement/*.'+filetype)

# Initialises the dose and channel varaibles
doseRef = array([])
opticalDensityVals = array([[],[],[]])


for i in range(numFiles):
	
	# Loads the current calibration file
	im = imread(calibrationFiles[i])
	
	# Pulls the defined dose from the file name
	currentDose = float(pull_filename(calibrationFiles[i]))
	
	# Defines a pixel grid over the image
	dim = shape(im[:,:,0])
	y, x = mgrid[0:dim[0], 0:dim[1]]
	
	# Converts from pixels into cm
	x = x * pixel2dist
	y = y * pixel2dist
	
	# Using predefined margins defines the valid internal rectangle of this image
	center = ( 
		(x > x.min() + xmargin) & 
		(x < x.max() - xmargin) & 
		(y > y.min() + ymargin) & 
		(y < y.max() - ymargin)
	)
	
	# Pulls out the valid pixel values from the image
	opticalDensityRed = -log(im[:,:,0][center])
	opticalDensityGreen = -log(im[:,:,1][center])
	opticalDensityBlue = -log(im[:,:,2][center])
	
	# Stores the pixel values for each channel along with the reference dose
	doseRef = append(doseRef,currentDose * ones(shape(opticalDensityRed)),axis=1)
	opticalDensityVals = append(opticalDensityVals,[opticalDensityRed,opticalDensityGreen,opticalDensityBlue],axis=1)
	

# Using least squares fitting calculates the parameters for each of the colour channels
red, redLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[0,:])
green, greenLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[1,:])
blue, blueLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[2,:])

# Defines the xi values for plotting fits
dosei = linspace(doseRef.min(),doseRef.max(),100)


figure(1)
clf()

plot(doseRef,opticalDensityVals[0,:], 'r.')
plot(dosei,dose2PixelVal(dosei, red), 'r-')

plot(doseRef,opticalDensityVals[1,:], 'g.')
plot(dosei,dose2PixelVal(dosei, green), 'g-')

plot(doseRef,opticalDensityVals[2,:], 'b.')
plot(dosei,dose2PixelVal(dosei, blue), 'b-')


show()


# ========================= #
#      Experimentation      #
# ========================= #

# im = imread(measurementFiles[0])

# pixelVals = 1 - im[100,100,:]

# doseInitGuess = pixelVal2Dose(pixelVals[0],red)


# def to_be_minimised(thick,red,green,blue,pixelVals):
	


	
# basinhopping(to_be_minimised,thick0)

# pixelVali = linspace(0.3,0.6,100)
# plot(pixelVal2Dose(pixelVali, red),pixelVali,'x')


	
