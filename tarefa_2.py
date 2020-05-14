import numpy as np
import matplotlib.pyplot as plt
import time
import restthru

host = 'http://127.0.0.1:4950'
restthru.http_init()

Ks = 0.5
axis = 165
R_sigma_x = 0.5
R_sigma_y = 0.5
R_sigma_th = 0.03
timespan = 3

def calculateEllipse(x,y,a,b,angle,steps=36):
    beta = -angle*np.pi/180
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
  
R = np.matrix([[R_sigma_x,0,0],[0,R_sigma_y,0],[0,0,R_sigma_th]])

l_old = 0
r_old = 0
th_kinematic_old = 0
s = 0
sigma_p_old = np.zeros([3,3])
x_kinematic, y_kinematic, th_kinematic = getPose()
kinematic  = np.array([x_kinematic,y_kinematic,th_kinematic])


i = 0
plt.ion()
fig, ax = plt.subplots()

while(i < 100):
    l_vel,r_vel = getVel()

    l = l_vel * timespan
    r = r_vel * timespan

    l = l + l_old
    r = r + r_old

    delta_l = l - l_old
    delta_r = r - r_old

    delta_th = (delta_r + delta_l)/(2*axis)

    if delta_th > np.pi:
        delta_th = delta_th-2*np.pi
    elif delta_th < - np.pi:
        delta_th = delta_th+2*np.pi

    delta_s = (delta_l + delta_r)/2 

    nabla_p_f = np.matrix([[1, 0, -delta_s*np.sin(th_kinematic+delta_th/2)], [0, 1, delta_s*np.cos(th_kinematic+delta_th/2)], [0, 0, 1]])

    nabla_s_f = np.matrix([[1/2*np.cos(th_kinematic+delta_th/2) - delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2), 1/2*np.cos(th_kinematic+delta_th/2) + delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2)], [1/2*np.sin(th_kinematic+delta_th/2) + delta_s/(4*axis)*np.cos(th_kinematic+delta_th/2), 1/2*np.sin(th_kinematic+delta_th/2) - delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2)], [1/(2*axis), -1/(2*axis)]])

    sigma_delta_s = np.matrix([[Ks * np.abs(delta_r),0],[0,Ks * np.abs(delta_l)]])

    sigma_p = np.matrix(np.dot(np.dot(nabla_p_f,sigma_p_old),nabla_p_f.T) + np.dot(np.dot(nabla_s_f,sigma_delta_s),nabla_s_f.T)) + R
    
    print(np.linalg.det(sigma_p))
    
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

    kinematic_next =  np.array([(delta_r+delta_l)/2*np.cos(th_kinematic+(delta_r-delta_l)/(4*axis)),(delta_r+delta_l)/2*np.sin(th_kinematic+(delta_r-delta_l)/(4*axis)),(delta_r-delta_l)/(2*axis)])
    kinematic = kinematic + kinematic_next

    x_kinematic = kinematic[0]
    y_kinematic = kinematic[1]
    th_kinematic = kinematic[2]


    X,Y = calculateEllipse(x_kinematic,y_kinematic,a,b,beta)
    ax.quiver(x_kinematic,y_kinematic,np.cos(th_kinematic),np.sin(th_kinematic),width=0.0005)
    ax.scatter(x_kinematic,y_kinematic,color='r')
    ax.plot(X,Y)

    x_odometry,y_odometry,th_odometry = getPose()
    ax.quiver(x_odometry,y_odometry,np.cos(th_odometry),np.sin(th_odometry),width=0.0005)
    ax.scatter(x_odometry,y_odometry,color='b')

    plt.pause(timespan)
    plt.draw()
    i += 1

    l_old = l
    r_old = r
    th_old = th_kinematic
    sigma_p_old = sigma_p



