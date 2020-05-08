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

def getPose():
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

def getVel():
    vel2 = "/motion/vel2"
    res,_ = restthru.http_get(host+vel2)
    l = res["left"]
    r = res["right"]

    return l,r
  
i = 0
plt.ion()
fig, ax = plt.subplots()

nabla_p_f = np.matrix([[1, 0, -delta_s*np.sin(th+delta_th/2)], [0, 1, delta_s*np.cos(th+th/2)], [0, 0, 1]])

nabla_s_f = np.matrix([[1/2*np.cos(th+delta_th/2) - delta_s/(4*b)*np.sin(th+delta_th/2), 1/2*np.cos(th+delta_th/2) + delta_s/(4*b)*np.sin(th+delta_th/2)], [1/2*np.sin(th+delta_th/2) - delta_s/(4*b)*np.cos(th+delta_th/2), 1/2*np.sin(th+delta_th/2) + delta_s/(4*b)*np.cos(th+delta_th/2)], [1/(2*b), -1/(2*b)]])

sigma_delta_s = np.matrix([[Ks * np.abs(delta_d),0],[0,Ks * np.abs(delta_e)]])

sigma_p = 0

sigma_p_new = np.dot(np.dot(nabla_p_f,sigma_p),nabla_p_f.T) + np.dot(np.dot(nabla_s_f,sigma_delta_s),nabla_s_f.T) 


while(i < 10):
    x,y,pth = getPose()
    X,Y = calculateEllipse(x,y,10,20,30)
    ax.quiver(x,y,np.cos(pth),np.sin(pth))
    ax.scatter(x,y)
    ax.plot(X,Y)
    plt.pause(3)
    plt.draw()
    i += 1




