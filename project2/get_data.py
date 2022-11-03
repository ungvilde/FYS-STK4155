import numpy as np
import matplotlib.pyplot as plt
from common import FrankeFunction

def polynomial(x):
    return 0.5 + 0.5*x + 5*x**2 - 4*x**3

np.random.seed(2022)

n = 1000 # num. data points
x = np.random.rand(n)
sigma = 0.1
eps = sigma*np.random.randn(n)
y = polynomial(x) + eps

plt.plot(x, y, '.')
plt.xlabel("$x$ values")
plt.ylabel("$y$ values")
plt.show()

filename = f"datasets/Poly_degree3_sigma{sigma}_N{n}.txt"

with open(filename, "w") as outfile:
    for i in range(n):
        outfile.write(str(x[i]) + " " + str(y[i]))
        outfile.write("\n")

# Make Franke data
n=100
x = np.random.rand(n)
y = np.random.rand(n)
x, y = np.meshgrid(x, y)

sigma = 0.01
eps = sigma*np.random.randn(n)
z = FrankeFunction(x, y) + eps

x = x.ravel()
y = y.ravel()
z = z.ravel()

filename = f"datasets/Franke_sigma{sigma}_N{n*n}.txt"

with open(filename, "w") as outfile:
    for i in range(n*n):
        outfile.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]))
        outfile.write("\n")