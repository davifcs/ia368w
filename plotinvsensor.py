import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

Rmax = 10
Amax = 5.0
d = 5
mu = np.array([d,0])

sigma = np.array([[0.5,0],[0,0.05]])

invSigma = np.linalg.inv(sigma)
Pmin = 0.1
K = 0.02

x = np.linspace(0 , Rmax, 400)
y = np.linspace(-Amax, Amax, 400)
z = np.empty([len(y),len(x)])
xx, yy = np.meshgrid(x, y)
max = 0

for i in range(len(x)):
    for j in range(len(y)):
        if x[i] < mu[0]:
            P = Pmin
        else:
            P = 0.5
        X  = np.array([x[i],y[j]])
        z[j][i] = P + (K/(2*np.pi*np.dot(sigma[0][0],sigma[1][1])) + 0.5 - P)*np.exp(-.5 * (np.dot(np.dot((X - mu),invSigma),(X - mu).T)))
        if z[j][i] > max:
            max = z[j][i]


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()