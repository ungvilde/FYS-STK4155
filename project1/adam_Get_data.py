import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform

# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')
print(np.shape(terrain))
N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
#plt.imshow(terrain, cmap='gray')
plt.contourf(x_mesh, y_mesh, z, cmap='gray')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

from LinearRegression import LinearRegression
#from Resample import Resample

ols = LinearRegression(20, x_mesh, y_mesh, z)
lasso = LinearRegression(20, x_mesh, y_mesh, z, lmbd=0.01)
ridge = LinearRegression(20, x_mesh, y_mesh, z, lmbd=0.01)

#X_test, z_test = ols.split_predict_eval(fit=False)

plt.figure()
plt.title('Terrain over Norway 1 OLS')
#plt.imshow(terrain, cmap='gray')
plt.contourf(ols(), cmap='gray')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#X_test, z_test = lasso.split_predict_eval(fit=False)

plt.figure()
plt.title('Terrain over Norway 1 LASSO')
#plt.imshow(terrain, cmap='gray')
plt.contourf(lasso(), cmap='gray')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#X_test, z_test = ridge.split_predict_eval(fit=False)

plt.figure()
plt.title('Terrain over Norway 1 Ridge')
#plt.imshow(terrain, cmap='gray')
plt.contourf(ridge(), cmap='gray')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


'''
ols_res = Resample(ols)
lasso_res = Resample(lasso)
ridge_res = Resample(ridge)

ols_boot = ols_res.bootstrap(30)
lasso_boot = lasso_res.bootstrap(30)
ridge_boot = ridge_res.bootstrap(30)

ols_cv = ols_res.cross_validation(10)
lasso_cv = lasso_res.cross_validation(10)
ridge_cv = ridge_res.cross_validation(10)
'''