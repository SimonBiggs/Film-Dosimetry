from matplotlib.pyplot import *
from matplotlib.image import *

from numpy import *
from scipy.optimize import curve_fit

from glob import glob
import os

from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed, RotatingMarker

from film_functions import fitting_func, dose2density, save_dict


def polynomial_4thOrder(x,a,b,c,d,e):
	return a*x**4 + b*x**3 + c*x**2 + d*x + e


# Defines the imageset
imageset = '005_Uniformity_trial'

# Pulls the calibration file paths names
uniformityImageFilepaths = glob('../image_sets/'+imageset+'/*.png')
numFiles = shape(uniformityImageFilepaths)[0]

im0 = imread(uniformityImageFilepaths[0])
dim = shape(im0)[0:2]

keys = [0,1,2,3,4,5]
xTest = [500,600,700,800,900,1000]
xRange = {
	'0':arange(500,550),
	'1':arange(600,630),
	'2':arange(700,740),
	'3':arange(800,830),
	'4':arange(900,930),
	'5':arange(1000,1030),
}

yValStore = {}
PValStore = {}

for i in range(len(keys)):
	yValStore[str(keys[i])] = array([])
	PValStore[str(keys[i])] = array([])

for i in range(numFiles):

	im = imread(uniformityImageFilepaths[i])
	
	for j in range(len(keys)):
		testVal = xTest[j]
		testStrip = im[:,testVal,0]

		y = arange(len(testStrip))
		yRef = y[(y>=10) & (y<len(y)-10) & (testStrip != 1)][20:-20]

		xRef = xRange[str(keys[j])]

		xRefGrid, yRefGrid = meshgrid(xRef,yRef)
		testIm = im[yRefGrid,xRefGrid,0]

		yValStore[str(keys[j])] = append(yValStore[str(keys[j])],ravel(yRefGrid))
		PValStore[str(keys[j])] = append(PValStore[str(keys[j])],ravel(testIm))



# for i in range(len(keys)):
	# figure(i+1)
	# plot(yValStore[str(keys[i])],PValStore[str(keys[i])],'.')


# show()


fit = {}
cov = {}

for i in range(len(keys)):
	fit[str(keys[i])], cov[str(keys[i])] = curve_fit(polynomial_4thOrder,yValStore[str(keys[i])],PValStore[str(keys[i])])


clf()
close("all")

yi = linspace(30,dim[0]-30)
for i in range(len(keys)):
	fitRet = polynomial_4thOrder(yi,*fit[str(keys[i])])
	plot(yi,fitRet/fitRet.max())

show()