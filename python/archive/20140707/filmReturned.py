#!/usr/bin/env python

# # Defines the file to look in for the images
# imageset = '001_Matt_brachy'

# # File type of the images
# filetype = 'png'

# # The pixel to cm conversion. The images in question use 70dpi
# pixel2dist = 2.54/70

# # Defines margin to be removed on the film measurement
# xmargin = 2.
# ymargin = 0.5


# # Use this and the above code to make an input file
# data = {"imageset": imageset, "filetype": filetype, 
	# "pixel2dist": pixel2dist, 
	# "xmargin": xmargin, "ymargin": ymargin}

# import json
# with open('../image_sets/'+imageset+'/inputdata', 'w') as outfile:
  # json.dump(data, outfile, sort_keys=True, indent=4, separators=(',', ': '))


from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *
from matplotlib.image import *
from numpy import *
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import basinhopping, curve_fit, minimize
from scipy.signal import wiener
from glob import glob
import os
import json
import time
# For the following to work inside ipython need to run the following in cmd.exe:
# pip install progressbar-ipython
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed, RotatingMarker
from scipy.stats import norm


def fitting_func(x, a, b, c):
	# The form for the fitting -- From Micke (2011)
  return -log((a + b*x)/(c + x))

def dose2density(dose, param):
	# Easier use with the output of curve_fit
	OD = fitting_func(dose, param[0], param[1], param[2])
	return OD
	
def density2Dose(OD, param):
	# The inverse function of the "fitting_func"
	# \frac{c e^{-OD} - a}{b - e^{-OD}}
	dose = (param[2] * exp(-OD) - param[0]) / (param[1] - exp(-OD))
	return dose

def density2DoseSigma(OD, param, paramSig):
	# Uncertainty of the numerator: c e^{-OD} - a 
	sigNumerator = sqrt((paramSig[2] * exp(-OD))**2 + paramSig[0]**2)
	numerator = param[2] * exp(-OD) - param[0]
	
	# Uncertainty of the denominator: b - e^{-OD}
	sigDenominator = paramSig[1]
	denominator = param[1] - exp(-OD)
	
	# Final uncertainty
	sigTotal = sqrt((sigNumerator/numerator)**2 + (sigDenominator/denominator)**2) * numerator / denominator
	return sigTotal
	

def pull_filename(fullPath):
	# Converts full path name into just the file name with extension removed
	# Used to make titles and to pull Dose value out of file names
	filename = os.path.basename(fullPath)
	result = filename[0:len(filename)-4]
	return result


def to_be_minimised(T,OD,w,red,green,blue,pixelCalcWidth):
	D = zeros([pixelCalcWidth,pixelCalcWidth,3])
	
	D[:,:,0] = density2Dose(OD[:,:,0] / T,red)
	D[:,:,1] = density2Dose(OD[:,:,1] / T,green)
	D[:,:,2] = density2Dose(OD[:,:,2] / T,blue)
	
	Davg = sum(D*w)/sum(w)

	return sum(w*(D - Davg)**2)
	
# def weighted_median(x,w):
	# ref = argsort(x)
	# xSort = x[ref]
	# wSort = w[ref]
	
	# mid = sum(w)/2
	# xSortH1 = xSort[cumsum(wSort)<mid]
	
	# xMed = xSortH1[len(xSortH1)-1]
	# return xMed
	

	
# def to_be_minimised(T,OD,w,red,green,blue,pixelCalcWidth):
	# D = zeros([pixelCalcWidth,pixelCalcWidth,3])
	
	# D[:,:,0] = density2Dose(OD[:,:,0] / T,red)
	# D[:,:,1] = density2Dose(OD[:,:,1] / T,green)
	# D[:,:,2] = density2Dose(OD[:,:,2] / T,blue)
	
	# variance = std(D)**2
	
	# pixelMeanDose = (w[:,:,0]*D[:,:,0] + w[:,:,1]*D[:,:,1] + w[:,:,2]*D[:,:,2]) / (w[:,:,0] + w[:,:,1] + w[:,:,2] )

	# pixelDeviations = (w[:,:,0]*(D[:,:,0] - pixelMeanDose)**2 + w[:,:,1]*(D[:,:,1] - pixelMeanDose)**2 + w[:,:,2]*(D[:,:,2] - pixelMeanDose)**2) / (w[:,:,0] + w[:,:,1] + w[:,:,2] )

	# valid = (pixelDeviations < variance) & (pixelDeviations < 200)
	
	# D2 = D[valid]
	# w2 = w[valid]
	
	# # Make this only use valid points?
	# # Dmed = weighted_median(D2,w2)
	
	# Davg = sum(D[valid]*w[valid])/sum(w[valid])
	
	
	# # diffsSqrd = (D2 - Dmed)**2
	# # diffsSqrdMed = weighted_median(diffsSqrd,w2)
	
	# return sum(w[valid]*(D[valid] - Davg)**2)/sum(w[valid])
	

def doseAverager(T,OD,w,red,green,blue,pixelCalcWidth):
	D = zeros([pixelCalcWidth,pixelCalcWidth,3])
	
	D[:,:,0] = density2Dose(OD[:,:,0] / T,red)
	D[:,:,1] = density2Dose(OD[:,:,1] / T,green)
	D[:,:,2] = density2Dose(OD[:,:,2] / T,blue)
	
	Davg = sum(D*w)/sum(w)
	
	return Davg
	

# def doseMedian(T,OD,w,red,green,blue,pixelCalcWidth):
	# D = zeros([pixelCalcWidth,pixelCalcWidth,3])
	
	# D[:,:,0] = density2Dose(OD[:,:,0] / T,red)
	# D[:,:,1] = density2Dose(OD[:,:,1] / T,green)
	# D[:,:,2] = density2Dose(OD[:,:,2] / T,blue)
	
	# D2 = reshape(D,-1)
	# w2 = reshape(w,-1)
	
	# Dmed = weighted_median(D2,w2)
	
	# return Dmed


	
# ===================

# Defines the imageset
imageset = '003_Matt_brachy_aged'

# Loads the input data file
with open('../image_sets/'+imageset+'/inputdata', 'r') as infile:
	data = json.load(infile)

# Converts data file values into required variables
filetype = data['filetype']
pixel2dist = data['pixel2dist']
xmargin = data['xmargin']
ymargin = data['ymargin']

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
		(x >= x.min() + xmargin) & 
		(x <= x.max() - xmargin) & 
		(y >= y.min() + ymargin) & 
		(y <= y.max() - ymargin)
	)
	
	invalid = (
		(im[:,:,0] == 0) & 
		(im[:,:,1] == 0) & 
		(im[:,:,2] == 0)
	)
	
	# Pulls out the valid pixel values from the image
	opticalDensityRed = -log(im[:,:,0][center & ~invalid])
	opticalDensityGreen = -log(im[:,:,1][center & ~invalid])
	opticalDensityBlue = -log(im[:,:,2][center & ~invalid])
	
	# Stores the pixel values for each channel along with the reference dose
	doseRef = append(doseRef,currentDose * ones(shape(opticalDensityRed)),axis=1)
	opticalDensityVals = append(opticalDensityVals,[opticalDensityRed,opticalDensityGreen,opticalDensityBlue],axis=1)
	

# Using least squares fitting calculates the parameters for each of the colour channels
red, redLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[0,:])
redSig = sqrt(diag(redLsqCov))

green, greenLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[1,:])
greenSig = sqrt(diag(greenLsqCov))

blue, blueLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[2,:])
blueSig = sqrt(diag(blueLsqCov))

with open('../image_sets/'+imageset+'/calibrationdata.npz', 'w') as outfile:
  savez(outfile, red=red, green=green, blue=blue, 
		redSig=redSig, greenSig=greenSig, blueSig=blueSig)
	

# ===================
	
	
	
	
# # Defines the xi values for plotting fits
# dosei = linspace(doseRef.min(),doseRef.max(),100)


# figure(1)
# clf()

# plot(doseRef,opticalDensityVals[0,:], 'r.')
# plot(dosei,dose2density(dosei, red), 'r-')

# plot(doseRef,opticalDensityVals[1,:], 'g.')
# plot(dosei,dose2density(dosei, green), 'g-')

# plot(doseRef,opticalDensityVals[2,:], 'b.')
# plot(dosei,dose2density(dosei, blue), 'b-')


# show()
# ========================= #
#      Experimentation      #
# ========================= #

im = imread(measurementFiles[0])
#im = wiener(im)
dim = shape(im)

densityVals = -log(im[20:dim[0]-20:1,20:dim[1]-20:1,:])
dim = shape(densityVals)

pixelCalcWidth = 20;

T = zeros([dim[0]/pixelCalcWidth,dim[1]/pixelCalcWidth])
D = zeros([dim[0]/pixelCalcWidth,dim[1]/pixelCalcWidth])
w = zeros([pixelCalcWidth,pixelCalcWidth,3])
OD = zeros([pixelCalcWidth,pixelCalcWidth,3])


widgets = ['Dose calc: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets = widgets, maxval=dim[0]/pixelCalcWidth).start() 



for i in range(dim[0]/pixelCalcWidth):
	idist = i*pixelCalcWidth
	
	for j in range(dim[1]/pixelCalcWidth):  
		jdist = j*pixelCalcWidth
		
		OD = densityVals[idist:idist+pixelCalcWidth, jdist:jdist+pixelCalcWidth, :]
		w[:,:,0] = 1/density2DoseSigma(OD[:,:,0],red,redSig)
		w[:,:,1] = 1/density2DoseSigma(OD[:,:,1],green,greenSig)
		w[:,:,2] = 1/density2DoseSigma(OD[:,:,2],blue,blueSig)
		
		ret = minimize(to_be_minimised, 1, args=(OD,w,red,green,blue,pixelCalcWidth))
		
		T[i,j] = ret.x
		D[i,j] = doseAverager(T[i,j],OD,w,red,green,blue,pixelCalcWidth)
	
	pbar.update(i+1)
	
pbar.finish()


dim = shape(D)

y, x = mgrid[0:dim[0], 0:dim[1]]
x = x * pixel2dist * pixelCalcWidth
y = y * pixel2dist * pixelCalcWidth


with open('../image_sets/'+imageset+'/saveddata2.npz', 'w') as outfile:
  savez(outfile, x=x, y=y, D=D, T=T)
	
# with open('../image_sets/'+imageset+'/saveddata.npz', 'r') as infile:
	# npzfile = load(infile)
	
# x = npzfile['x']
# y = npzfile['y']
# D = npzfile['D']
# T = npzfile['T']


# fig2 = figure(2)
# clf()
# ax2 = fig2.gca(projection='3d')

# ax2.plot(reshape(x,-1), reshape(y,-1), reshape(D,-1),'.')


# fig3 = figure(3)
# clf()
# ax3 = fig3.gca(projection='3d')

# ax3.plot(reshape(x,-1), reshape(y,-1), reshape(T,-1),'.')


# fig5 = figure(5)
# clf()
# doseJustRed = density2Dose(densityVals[:,:,0],red)

# pic2 = imshow(doseJustRed, cmap=cm.jet, vmin=D.min(), vmax=D.max(), interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig5.colorbar(pic2)


fig11 = figure(11)
clf()
pic11 = imshow(D, cmap=cm.jet, vmin=220, vmax=D.max(), interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
fig11.colorbar(pic11)



# fig12 = figure(12)
# clf()
# pic12 = imshow(Dfilt, cmap=cm.jet, vmin=500, vmax=1350, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig12.colorbar(pic12)




fig13 = figure(13)
clf()
pic13 = imshow(T, cmap=cm.jet, vmin=0.99, vmax=1.01, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
fig13.colorbar(pic13)

show()

# splineWeight = norm.pdf(log(T),0,std(log(T)))

# splineWeight = abs(log(T) / std(log(T)))

# splineX = ravel(x)
# splineY = ravel(y)
# splineD = ravel(D)



# doseSpline = SmoothBivariateSpline(splineY,splineX,splineD,w=ravel(splineWeight))


# zi = doseSpline.ev(y,x)

# fig16 = figure(16)
# clf()
# pic16 = imshow(zi, cmap=cm.jet, vmin=zi.min(), vmax=zi.max(), interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig16.colorbar(pic16)


# fig14 = figure(14)
# clf()
# pic14 = imshow(splineWeight, cmap=cm.jet, vmin=splineWeight.min(), vmax=splineWeight.max(), interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig14.colorbar(pic14)

# fig15 = figure(15)
# clf()
# pic15 = imshow(log(T), cmap=cm.jet, vmin=-0.2, vmax=0.2, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig15.colorbar(pic15)






# fig6 = figure(6)
# clf()
# diffImg = D - doseJustRed

# pic3 = imshow(diffImg, cmap=cm.jet, vmin=diffImg.min(), vmax=diffImg.max(), interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig6.colorbar(pic3)


# fig7 = figure(7)
# clf()
# pic4 = imshow(T, cmap=cm.jet, vmin=0.8, vmax=1.2, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig7.colorbar(pic4)


# show()





## END ##


	
	
# fig2 = figure(2)
# clf()
# ax2 = fig2.gca(projection='3d')

# ax2.plot_wireframe(x, y, T)


# thicknessSpline = RectBivariateSpline(y[:,0],x[0,:],T, s=0.07)

# xi = linspace(x.min(),x.max(),200)
# yi = linspace(y.min(),y.max(),200)

# yigrid, xigrid = meshgrid(yi, xi)


# fig4 = figure(4)
# clf()
# ax4 = fig4.gca(projection='3d')
# ax4.plot_surface(yigrid, xigrid, thicknessSpline.ev(yigrid, xigrid), alpha=0.3,rstride=4, cstride=4)

# show()






# T = ret.x
# Dred = density2Dose(OD[0] / T,red)
# Dgreen = density2Dose(OD[1] / T,green)
# Dblue = density2Dose(OD[2] / T,blue)

# print "Thickness =", T
# print "Colour doses:", Dred, Dgreen, Dblue
# print "Dose interpret =", (Dred + Dgreen + Dblue) / 3
# print " "
# basinhopping(to_be_minimised,thick0)

# pixelVali = linspace(0.3,0.6,100)
# plot(pixelVal2Dose(pixelVali, red),pixelVali,'x')

# ret = basinhopping(to_be_minimised, 1, minimizer_kwargs={"args": (OD,red,green,blue)})

	
