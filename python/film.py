# Defines the file to look in for the images
imageset = '001_Matt_brachy'

# File type of the images
filetype = 'png'

# The pixel to cm conversion. The images in question use 70dpi
pixel2dist = 2.54/70

# Defines margin to be removed on the film measurement
xmargin = 2
ymargin = 0.5


from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *
from matplotlib.image import *
from numpy import *
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import basinhopping, curve_fit, minimize
from glob import glob
import os


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
redSig = sqrt(diag(redLsqCov))

green, greenLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[1,:])
greenSig = sqrt(diag(greenLsqCov))

blue, blueLsqCov = curve_fit(fitting_func,doseRef,opticalDensityVals[2,:])
blueSig = sqrt(diag(blueLsqCov))

# Defines the xi values for plotting fits
dosei = linspace(doseRef.min(),doseRef.max(),100)


figure(1)
clf()

plot(doseRef,opticalDensityVals[0,:], 'r.')
plot(dosei,dose2density(dosei, red), 'r-')

plot(doseRef,opticalDensityVals[1,:], 'g.')
plot(dosei,dose2density(dosei, green), 'g-')

plot(doseRef,opticalDensityVals[2,:], 'b.')
plot(dosei,dose2density(dosei, blue), 'b-')



# ========================= #
#      Experimentation      #
# ========================= #

def to_be_minimised(T,OD,w,red,green,blue):
	Dred = density2Dose(OD[0] / T,red)
	Dgreen = density2Dose(OD[1] / T,green)
	Dblue = density2Dose(OD[2] / T,blue)
	
	Davg = (w[0]*Dred+w[1]*Dgreen+w[2]*Dblue)/sum(w)

	return w[0]*(Dred - Davg)**2 + w[1]*(Dgreen - Davg)**2 + w[2]*(Dblue - Davg)**2

def doseAverager(T,OD,w,red,green,blue):
	Dred = density2Dose(OD[0] / T,red)
	Dgreen = density2Dose(OD[1] / T,green)
	Dblue = density2Dose(OD[2] / T,blue)
	
	Davg = (w[0]*Dred+w[1]*Dgreen+w[2]*Dblue)/sum(w)
	return Davg
	
im = imread(measurementFiles[0])
dim = shape(im)

densityVals = -log(im[20:dim[0]-20,20:dim[1]-20,:])
dim = shape(densityVals)

T = zeros([dim[0],dim[1]])
D = zeros([dim[0],dim[1]])
w = zeros(3)

for i in range(dim[0]):
	print float(i)/dim[0]*100,"%"
	for j in range(dim[1]):
		
		OD = densityVals[i,j,:]
		w[0] = 1/density2DoseSigma(OD[0],red,redSig)
		w[1] = 1/density2DoseSigma(OD[1],green,greenSig)
		w[2] = 1/density2DoseSigma(OD[2],blue,blueSig)
		ret = minimize(to_be_minimised, 1, args=(OD,w,red,green,blue))
		T[i,j] = ret.x
		D[i,j] = doseAverager(T[i,j],OD,w,red,green,blue)
	

	
y, x = mgrid[0:dim[0], 0:dim[1]]
x = x * pixel2dist
y = y * pixel2dist



fig2 = figure(2)
clf()
ax2 = fig2.gca(projection='3d')

ax2.plot(reshape(x,-1), reshape(y,-1), reshape(D,-1),'.')


fig3 = figure(3)
clf()
ax3 = fig3.gca(projection='3d')

ax3.plot(reshape(x,-1), reshape(y,-1), reshape(T,-1),'.')


show()

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

	
