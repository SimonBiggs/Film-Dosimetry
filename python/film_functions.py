from numpy import *

import os
from glob import glob

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
	

def to_be_minimised(T,ODr,ODg,ODb,fitR,fitG,fitB,w):
	D = zeros(3)
	
	D[0] = density2Dose(ODr / T,fitR)
	D[1] = density2Dose(ODg / T,fitG)
	D[2] = density2Dose(ODb / T,fitB)
	
	Davg = sum(D*w)/sum(w)

	return sum(w*(D - Davg)**2)


def doseAverager(T,ODr,ODg,ODb,fitR,fitG,fitB,w):
	D = zeros(3)
	
	D[0] = density2Dose(ODr / T,fitR)
	D[1] = density2Dose(ODg / T,fitG)
	D[2] = density2Dose(ODb / T,fitB)
	
	Davg = sum(D*w)/sum(w)
	
	return Davg
	
	
def pull_filename(fullPath):
	# Converts full path name into just the file name with extension removed
	filename = os.path.basename(fullPath)
	result = filename[0:len(filename)-4]
	return result
	

def load_dict(directory):
	loadDict = {}
	loadFiles = glob(directory+'/*.csv')
	
	for i in range(len(loadFiles)):
		filepath = loadFiles[i]
		currKey = pull_filename(filepath)
		loadDict[currKey] = loadtxt(filepath, delimiter=",")

	return loadDict
	
	
def save_dict(saveDict,directory):

	for i in range(len(saveDict)):
		currKey = saveDict.keys()[i]
		filename = directory+'/'+currKey+'.csv'
		savetxt(filename, saveDict[currKey], delimiter=",")