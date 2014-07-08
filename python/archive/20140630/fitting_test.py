import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
  return -np.log((a + b*x)/(c + x))

def func_output(x, param):
	return func(x, param[0], param[1], param[2])

xdata = np.linspace(1, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.01 * np.random.normal(size=len(xdata))

popt, pcov = curve_fit(func, xdata, ydata)

plt.plot(xdata,ydata,'.')
plt.plot(xdata,func(xdata, popt[0], popt[1], popt[2]),'-')

plt.show()