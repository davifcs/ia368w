import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import restthru

host = 'http://127.0.0.1:4950'
# inicia API
restthru.http_init()
# obtem leituras do laser e da posicao
distances = '/group/laserpose'

def sensing():
    res,_ = restthru.http_get(host+distances)

    px = res[1]["pose"]["x"]
    py = res[1]["pose"]["y"]
    pth = res[1]["pose"]["th"]*np.pi/180
    dist = np.asarray(res[0]["distances"])
    
    fi = np.linspace(pth - np.pi/2, pth + np.pi/2, 181)

    for index,value in enumerate(fi):
        if value > np.pi:
            fi[index] = (fi[index]-2*np.pi)
        elif value < - np.pi:
            fi[index] = (fi[index]+2*np.pi)
    fi = np.asarray(fi)
    #leitura do sensor
    mu_array = np.vstack((dist, fi)).T
    return px, py, pth, mu_array

tamCel = 50

numCelX = 100
numCelY = 100 
mapa = np.zeros([numCelX,numCelY])

precisao = 50 
passo = 1*np.pi/180

sigma = np.array([[0.05,0],[0,0.005]])
invSigma = np.linalg.inv(sigma)
Pmin = 0.4

K = 0.0001

x = np.linspace(0, numCelX, numCelX)
y = np.linspace(0, numCelY, numCelY)
xx, yy = np.meshgrid(x, y)
maximo = 0

z = np.zeros([len(x),len(y)])

fig = plt.figure()
ax = fig.gca(projection='3d')

while(True):
    px, py, pth, mu_array = sensing()
    for i in range(len(x)):
        xi = (i-1)*tamCel+tamCel/2
        for j in range(len(y)):
            yj = (j-1)*tamCel + tamCel/2
            r = np.sqrt((xi-px)**2+(yj-py)**2)
            angCel = np.arctan2((yj-py),(xi-px))
            b = angCel - pth
            if b > np.pi:
                b = (b-2*np.pi)
            elif b < - np.pi:
                b = (b+2*np.pi)
            if abs(b) <= np.pi/2:
                for mu in mu_array:
                    delta = np.array([r, angCel]) - mu                
                    if abs(delta[1])>2*passo:
                        continue
                    if r < mu[0]:
                        P = Pmin
                    else:
                        P = 0.5
                    delta[0]  = delta[0]/1000
                    z[j][i] = P + (K/(2*np.pi*np.dot(sigma[0][0],sigma[1][1])) + 0.5 - P)*np.exp(-0.5 * (np.dot(np.dot(delta,invSigma),delta.T)))
                    if abs(mapa[j][i]) < 0.8:
                        mapa[j][i] = mapa[j][i] + np.log(z[j][i]/(1-z[j][i]))
                    if z[j][i] > maximo:
                        maximo = z[j][i]
    surf = ax.plot_surface(xx,yy,mapa, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.pause(0.0001)

