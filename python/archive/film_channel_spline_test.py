from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *
from matplotlib.image import *
from numpy import *
from scipy.interpolate import *
from scipy.optimize import curve_fit
from glob import glob
import os

def func(x, a, b, c):
  return -log((a + b*x)/(c + x))

def func_output(x, param):
	return func(x, param[0], param[1], param[2])

def pull_filename(fullPath):
	# Converts full path name into just the file name with extension removed
	# Used to make titles and to pull Dose value out of file names
	
	filename = os.path.basename(fullPath)
	result = filename[0:len(filename)-4]
	
	return result



imageset = '001_Matt_brachy'


calibrationFiles = glob('../image_sets/'+imageset+'/calibration/*.png')
numFiles = shape(calibrationFiles)[0]
# ax = range(numFiles)
# fig1 = figure(1)
# clf()

measurementFiles = glob('../image_sets/'+imageset+'/measurement/*.png')
im = imread(measurementFiles[0])

# axLeft = subplot2grid((int(ceil(numFiles/2)+1),3),(0,0),rowspan=int(ceil(numFiles/2)))

# title(pull_filename(measurementFiles[0]))

redDose = 1 - im[:,:,0];

dim = shape(redDose);
x, y = mgrid[0:dim[0], 0:dim[1]]

# colWash = axLeft.imshow(redDose, cmap=cm.jet, vmin=0, vmax=1, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])

calibrationDose = zeros(numFiles)
redMean = zeros(numFiles)
redStd = zeros(numFiles)
blueMean = zeros(numFiles)
blueStd = zeros(numFiles)
greenMean = zeros(numFiles)
greenStd = zeros(numFiles)

for i in range(numFiles):
	
	# ax[i] = subplot2grid( ( int(ceil(numFiles/2)+1), 3 ), (int(floor(i/2)), mod(i,2) + 1) )
	
	im = imread(calibrationFiles[i])
	
	calibrationDose[i] = float(pull_filename(calibrationFiles[i]))
	
	dim = shape(im[:,:,0])
	y, x = mgrid[0:dim[0], 0:dim[1]]

	x = x *2.54/70
	y = y * 2.54/70
	
	imred = zeros(shape(im))
	imblue = zeros(shape(im))
	imgreen = zeros(shape(im))

	imred[:,:,0] = im[:,:,0]
	imgreen[:,:,1] = im[:,:,1]
	imblue[:,:,2] = im[:,:,2]


	# Still need conversions before it is dose
	redDose = 1 - im[:,:,0]
	greenDose = 1 - im[:,:,1]
	blueDose = 1 - im[:,:,2]

	center = (x>x.min()+2) & (x<x.max()-2) & (y>y.min()+0.5) & (y<y.max()-0.5);
	
	redMean[i] = mean(redDose[center])
	redStd[i] = std(redDose[center])
	greenMean[i] = mean(greenDose[center])
	greenStd[i] = std(greenDose[center])
	blueMean[i] = mean(blueDose[center])
	blueStd[i] = std(blueDose[center])
	

	# ax[i].imshow(redDose, cmap=cm.jet, vmin=0, vmax=1, interpolation='none', extent=[x.min(),x.max(),y.max(),y.min()])
	# title(str(calibrationDose[i]) + ' Gy')


# fig1.colorbar(colWash, ax=axLeft)


# redSpline = UnivariateSpline(calibrationDose,redMean,w=1/redStd,s=5)
# greenSpline = UnivariateSpline(calibrationDose,greenMean,w=1/greenStd,s=5)
# blueSpline = UnivariateSpline(calibrationDose,blueMean,w=1/blueStd,s=5)

redLsqParam, redLsqCov = curve_fit(func,calibrationDose,redMean)
greenLsqParam, greenLsqCov = curve_fit(func,calibrationDose,greenMean)
blueLsqParam, blueLsqCov = curve_fit(func,calibrationDose,blueMean)

xi = linspace(calibrationDose.min(),calibrationDose.max(),100)

# fig2 = figure(2)
# clf()

# plot(calibrationDose,redMean, 'r.')
# plot(xi,redSpline(xi), 'r-')

# plot(calibrationDose,greenMean, 'g.')
# plot(xi,greenSpline(xi), 'g-')

# plot(calibrationDose,blueMean, 'b.')
# plot(xi,blueSpline(xi), 'b-')


fig2 = figure(3)
clf()

plot(calibrationDose,redMean, 'r.')
plot(xi,func_output(xi, redLsqParam), 'r-')

plot(calibrationDose,greenMean, 'g.')
plot(xi,func_output(xi, greenLsqParam), 'g-')

plot(calibrationDose,blueMean, 'b.')
plot(xi,func_output(xi, blueLsqParam), 'b-')




show()