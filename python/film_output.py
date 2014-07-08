from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *

from numpy import *

from glob import glob
import os

from film_functions import load_dict

from scipy.signal import cspline2d

# Defines the imageset
imageset = '004_Matt_brachy_aged_2'

# Defines pixel to cm conversion
pixel2dist = 0.02


# Pulls the measurement file paths names
measurementDirectories = glob('../image_sets/'+imageset+'/measurement/*')

# Load the measurement calculations data
imageName = os.path.basename(measurementDirectories[0])

measDict = load_dict('../image_sets/'+imageset+'/data/measurement/'+imageName)

D = measDict['D']
T = measDict['T']

# Load the calibration data
fit = load_dict('../image_sets/'+imageset+'/data/calibration')




dim = shape(D)

y, x = mgrid[0:dim[0], 0:dim[1]]
x = x * pixel2dist
y = y * pixel2dist






close("all")





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

#Are these axes correct??
fig11 = figure(11)
clf()
pic11 = imshow(D, cmap=cm.jet, vmin=D.min(), vmax=D.max(), interpolation='none', extent=[x.min(),x.max(),y.max(),y.min()])
fig11.colorbar(pic11)



# fig12 = figure(12)
# clf()
# pic12 = imshow(Dfilt, cmap=cm.jet, vmin=500, vmax=1350, interpolation='none', extent=[y.min(),y.max(),x.max(),x.min()])
# fig12.colorbar(pic12)




fig13 = figure(13)
clf()
pic13 = imshow(T, cmap=cm.jet, vmin=T.mean() - 3*T.std(), vmax=T.mean() + 3*T.std(), interpolation='none', extent=[x.min(),x.max(),y.max(),y.min()])
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

	
