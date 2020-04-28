import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import restthru

host = 'http://127.0.0.1:4950'
restthru.http_init()
distances = '/group/laserpose'

tamCel = 50
numCelX = 80
numCelY = 80

precisao = 50 
passo = 1*np.pi/180

K_sigma_array = np.array([[[0.5],[[2,0],[0,0.2]]],[[0.02],[[0.5,0],[0,0.05]]],[[0.0002],[[0.05,0],[0,0.005]]]])
Pmin = 0.4

x = np.linspace(0, numCelX, numCelX)
y = np.linspace(0, numCelY, numCelY)
xx, yy = np.meshgrid(x, y)

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
    mu_array = np.vstack((dist, fi)).T
    return px, py, pth, mu_array

for K, sigma in K_sigma_array:
    mapa = np.zeros([numCelX,numCelY])
    z = np.zeros([len(x),len(y)])
    maximo = 0
    
    fig = plt.figure(figsize=(18, 16))
    graph = 0
    start = time.time()
    
    invSigma = np.linalg.inv(sigma)
    
    print("K = ",K[0]) 
    print("sigma =", sigma)
    
    while(graph < 4):
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
                        mapa[j][i] = mapa[j][i] + np.log(z[j][i]/(1-z[j][i]))
                        if z[j][i] > maximo:
                            maximo = z[j][i]
        cycle = time.time()
        diff = int(cycle-start)
    
        if diff > 30:
            start = time.time()
            graph += 1
            ax = fig.add_subplot(2, 2, graph, projection='3d')   
            ax.plot_surface(xx,yy,mapa, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.view_init(azim=-45, elev=45)
    print("z máximo", maximo)
    fig.savefig(str(K[0]) + ".png") 