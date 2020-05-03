import numpy as np
import matplotlib.pyplot as plt
import time
import restthru

host = 'http://127.0.0.1:4950'
restthru.http_init()

def calculateEllipse(x,y,a,b,angle,steps=36):
    beta = -angle * np.pi/180
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)
    
    alpha = np.linspace(0, 360, steps)
    alpha = alpha * np.pi/180

    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta)
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta)

    return X,Y

def sensing():
    pose = "/motion/pose"
    res,_ = restthru.http_get(host+pose)

    px = res["x"]
    py = res["y"]
    pth = res["th"]*np.pi/180
    if pth > np.pi:
        pth = pth-2*np.pi
    elif pth < - np.pi:
        pth = pth+2*np.pi

    return px, py, pth

i = 0
plt.ion()
fig, ax = plt.subplots()

while(i < 10):
    x,y,pth = sensing()
    ax.quiver(x,y,np.cos(pth),np.sin(pth))
    ax.scatter(x,y)
    plt.pause(3)
    plt.draw()
    i += 1




