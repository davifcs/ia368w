import numpy as np
import matplotlib.pyplot as plt
import time
import restthru

host = 'http://127.0.0.1:4950'
restthru.http_init()

Ks = 0.05
b = 165
sigma_x = 0.5
sigma_y = 0.5
sigma_th = 0.03
timespan = 3

def calculateEllipse(x,y,a,b,angle,steps=36):
    beta = -angle
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
  
R = np.matrix([[sigma_x,0,0],[0,sigma_y,0],[0,0,sigma_th]])

l_old = 0
r_old = 0
th_old = 0
s = 0
sigma_p_old = np.zeros([3,3])

i = 0
plt.ion()
fig, ax = plt.subplots()

while(i < 10):
    x,y,th = getPose()
    l,r = getVel()

    delta_l = l - l_old
    delta_r = r - r_old
    delta_th = th - th_old

    delta_s = (delta_l + delta_r)/2 * timespan 

    nabla_p_f = np.matrix([[1, 0, -delta_s*np.sin(th+delta_th/2)], [0, 1, delta_s*np.cos(th+th/2)], [0, 0, 1]])

    nabla_s_f = np.matrix([[1/2*np.cos(th+delta_th/2) - delta_s/(4*b)*np.sin(th+delta_th/2), 1/2*np.cos(th+delta_th/2) + delta_s/(4*b)*np.sin(th+delta_th/2)], [1/2*np.sin(th+delta_th/2) - delta_s/(4*b)*np.cos(th+delta_th/2), 1/2*np.sin(th+delta_th/2) + delta_s/(4*b)*np.sin(th+delta_th/2)], [1/(2*b), -1/(2*b)]])

    sigma_delta_s = np.matrix([[Ks * np.abs(delta_r),0],[0,Ks * np.abs(delta_l)]])

    sigma_p = np.matrix(np.dot(np.dot(nabla_p_f,sigma_p_old),nabla_p_f.T) + np.dot(np.dot(nabla_s_f,sigma_delta_s),nabla_s_f.T))
    
    sigma_x = np.sqrt(sigma_p[0,0])
    sigma_y = np.sqrt(sigma_p[1,1])
    sigma_xy = sigma_p[1,0]
    a = np.sqrt(1/2*(sigma_x**2+sigma_y**2+np.sqrt((sigma_y**2-sigma_x**2)**2+4*sigma_xy**2)))
    b = np.sqrt(1/2*(sigma_x**2+sigma_y**2-np.sqrt((sigma_y**2-sigma_x**2)**2+4*sigma_xy**2)))

    beta = 1/2*(np.arctan(2*sigma_xy/(sigma_y**2-sigma_x**2)))
    if beta > np.pi:
        beta = beta-2*np.pi
    elif beta < - np.pi:
        beta = beta+2*np.pi

    
    print(beta)

    X,Y = calculateEllipse(x,y,a,b,beta)
    
    ax.quiver(x,y,np.cos(th),np.sin(th))
    
    ax.scatter(x,y)
    ax.plot(X,Y)
    plt.pause(timespan)
    plt.draw()
    i += 1

    l_old = l
    r_old = r
    th_old = th
    sigma_p_old = sigma_p



