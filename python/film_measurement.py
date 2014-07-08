from matplotlib.image import * #mpimg
from numpy import * #np

from scipy.optimize import minimize

from glob import glob
import os

# For the following to work inside ipython need to run the following in cmd.exe:
# pip install progressbar-ipython
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed, RotatingMarker

from film_functions import *

# Defines the imageset
imageset = '004_Matt_brachy_aged_2'

# Load the calibration data
fit = load_dict('../image_sets/'+imageset+'/data/calibration')


# Pulls the measurement file paths names
measurementDirectories = glob('../image_sets/'+imageset+'/measurement/*')


# Place holder for inserting a loop at a later date
i = 0

currentImages = glob(measurementDirectories[i]+'/*.png')
numImages = shape(currentImages)[0]

im0 = imread(currentImages[0])
dim = shape(im0)[0:2]

channel = {}
channel['red'] = zeros((dim[0],dim[1],numImages))
channel['green'] = zeros((dim[0],dim[1],numImages))
channel['blue'] = zeros((dim[0],dim[1],numImages))

for j in range(numImages):
	im = imread(currentImages[j])
	channel['red'][...,j] = im[...,0]
	channel['green'][...,j] = im[...,1]
	channel['blue'][...,j] = im[...,2]

# Should this averaging happen before the log?
OD = {}
OD['red'] = -log( mean(channel['red'],axis=2) )
OD['green'] = -log( mean(channel['green'],axis=2) )
OD['blue'] = -log( mean(channel['blue'],axis=2) )


T = zeros((dim[0],dim[1]))
D = zeros((dim[0],dim[1]))
w = zeros(3)


widgets = ['Dose calc: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets = widgets, maxval=dim[0]).start()


for i in range(dim[0]):
	
	for j in range(dim[1]):  
		
		# Not entirely correct calculating weighting outside of the minimisation
		# But the difference is minimal
		w[0] = 1/density2DoseSigma(OD['red'][i,j],fit['red'],fit['redSig'])
		w[1] = 1/density2DoseSigma(OD['green'][i,j],fit['green'],fit['greenSig'])
		w[2] = 1/density2DoseSigma(OD['blue'][i,j],fit['blue'],fit['blueSig'])
		
		ret = minimize ( to_be_minimised, 1, 
			args = (
				OD['red'][i,j],
				OD['green'][i,j],
				OD['blue'][i,j],
				fit['red'],fit['green'],fit['blue'],
				w
			)
		)
		
		T[i,j] = ret.x

		D[i,j] = doseAverager(
			T[i,j],
			OD['red'][i,j],
			OD['green'][i,j],
			OD['blue'][i,j],
			fit['red'],fit['green'],fit['blue'],
			w
		)
	
	save_dict(dict(D=D, T=T),'temp')
	
	pbar.update(i+1)

	
pbar.finish()


saveDict = dict(D=D, T=T)
imageName = os.path.basename(measurementDirectories[0])
directory = '../image_sets/'+imageset+'/data/measurement/'+imageName

save_dict(saveDict,directory)