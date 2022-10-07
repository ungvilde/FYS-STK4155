import numpy as np
from imageio import imread
import plotly.graph_objects as go

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

'''epsilon = 1e-2
den = [x if x > epsilon else 1 for x in np.std(z, axis=0)]
z = (z - np.mean(z, axis=0))/den'''

fig = go.Figure(data=[go.Surface(z=z, colorscale="jet", colorbar=dict(title="Height (m)"))])
fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
fig.update_layout(scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)"), autosize=False, scene_camera_eye=dict(x=1.8, y=1.2, z=1.1), width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
fig.write_image("test/terrain.pdf")

'''
from LinearRegression import LinearRegression
#from Resample import Resample

lasso = LinearRegression(5, x_mesh, y_mesh, z, lmbd=0.01)
ridge = LinearRegression(5, x_mesh, y_mesh, z, lmbd=0.01)
ols = LinearRegression(5, x_mesh, y_mesh, z)

#X_test, z_test = lasso.split_predict_eval(fit=False)

fig = go.Figure(data=[go.Surface(z=lasso(), colorscale="jet")])
fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
fig.update_layout(title="LASSO", autosize=False, scene_camera_eye=dict(x=1.8, y=1.2, z=1.1), width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
fig.write_image("test/lasso.pdf")


#X_test, z_test = ridge.split_predict_eval(fit=False)

fig = go.Figure(data=[go.Surface(z=ridge(), colorscale="jet")])
fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
fig.update_layout(title="Ridge", autosize=False, scene_camera_eye=dict(x=1.8, y=1.2, z=1.1), width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
fig.write_image("test/ridge.pdf")


#X_test, z_test = ols.split_predict_eval(fit=False)

fig = go.Figure(data=[go.Surface(z=ols(), colorscale="jet")])
fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
fig.update_layout(title="OLS", autosize=False, scene_camera_eye=dict(x=1.8, y=1.2, z=1.1), width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
fig.write_image("test/ols.pdf")
'''