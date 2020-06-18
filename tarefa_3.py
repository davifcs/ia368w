import numpy as np
import matplotlib.pyplot as plt
import time
import restthru
from FeatureDetection import FeatureDetection

host = 'http://127.0.0.1:4950'
restthru.http_init()

Ks = 0.05
axis = 165
R_sigma_x = 5
R_sigma_y = 5
R_sigma_th = 3*np.pi/180
timespan = 2

L_x = [0,  0,   920,  920,  4190,  4680, 4190,  4030, 4680]
L_y = [0,  2280, 2280, 3200, 2365,  2365, 3200,  0,    650]

L = np.array([L_x,L_y])

sigma_l_d = 0.5
sigma_l_theta = 0.1 * np.pi/180

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

def normAngle(angle):
    if angle > np.pi:
        angle = angle-2*np.pi
    elif angle < - np.pi:
        angle = angle+2*np.pi
    
    return angle

def getPose():
    pose = "/motion/pose"
    res,_ = restthru.http_get(host+pose)

    px = res["x"]
    py = res["y"]
    pth = res["th"]*np.pi/180
    pth = normAngle(pth)

    return px, py, pth

def getVel():
    vel2 = "/motion/vel2"
    res,_ = restthru.http_get(host+vel2)

    return res["left"], res["right"]
    
def getGlobalPose():
    global_pose = "/perception/laser/n/global_poses"
    res,_ = restthru.http_get(host+global_pose)

    return res

PoseR = getPose()
sigma_p_old = np.zeros([3,3])

R = np.matrix([[R_sigma_x,0,0],[0,R_sigma_y,0],[0,0,R_sigma_th]])

l_old = 0
r_old = 0
th_kinematic_old = 0
s = 0
sigma_p_old = np.zeros([3,3])
x_kinematic, y_kinematic, th_kinematic = getPose()
kinematic  = np.array([x_kinematic,y_kinematic,th_kinematic])

plt.ion()
fig, ax = plt.subplots()
plt.xlim(0,5000)
plt.ylim(0,5000)

while(True):    
    #Prediction step
    l_vel,r_vel = getVel()

    delta_s_l = l_vel * timespan
    delta_s_r = r_vel * timespan

    delta_th = (delta_s_r - delta_s_l)/(2*axis)

    delta_s_t = (delta_s_r + delta_s_l)/2

    delta_th = normAngle(delta_th)
    
    G_t = np.matrix([1,0,-delta_s_t*np.sin(theta_t_old+delta_theta_t/2)],[0,1,delta_s_t*np.cos(theta_t_old+delta_theta_t/2)],[0,0,1])

    a = 1/2*np.cos(theta_t_old+delta_theta_t/2)-delta_s_t/(4*b)*np.sin(theta_old+delta_theta_t/2)
    b = 1/2*np.sin(theta_t_old+delta_theta_t/2)+delta_s_t/(4*b)*np.cos(theta_old+delta_theta_t/2)
    c = 1/2*np.cos(theta_t_old+delta_theta_t/2)+delta_s_t/(4*b)*np.sin(theta_old+delta_theta_t/2)
    d = 1/2*np.sin(theta_t_old+delta_theta_t/2)-delta_s_t/(4*b)*np.cos(theta_old+delta_theta_t/2)

    V_t = np.matrix([a,b],[c,d],[1/(2*b),-1/(2*b)])

    # nabla_p_f = np.matrix([[1, 0, -delta_s*np.sin(th_kinematic+delta_th/2)], [0, 1, delta_s*np.cos(th_kinematic+delta_th/2)], [0, 0, 1]])

    # nabla_s_f = np.matrix([[1/2*np.cos(th_kinematic+delta_th/2) - delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2), 1/2*np.cos(th_kinematic+delta_th/2) + delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2)], [1/2*np.sin(th_kinematic+delta_th/2) + delta_s/(4*axis)*np.cos(th_kinematic+delta_th/2), 1/2*np.sin(th_kinematic+delta_th/2) - delta_s/(4*axis)*np.sin(th_kinematic+delta_th/2)], [1/(2*axis), -1/(2*axis)]])

    sigma_delta_t = np.matrix([[Ks * np.abs(delta_s_r),0],[0,Ks * np.abs(delta_s_l)]])

    sigma_t = np.matrix(np.dot(np.dot(G_t,sigma_p_old),G_t.T) + np.dot(np.dot(V_t,sigma_delta_t),V_t.T)) + R
    
    print(np.linalg.det(sigma_t))
    
    # Update step
    time.sleep(2)

    global_poses = getGlobalPose
    x_t, y_t, th_t = getPose()

    detected_x, detected_y, real_x, real_y, deltaBearing_array, z_range_array, z_bearing_array, Z_range_array, Z_bearing_array = FeatureDetection(global_poses, PoseR)
    
    H_t = 0
    Q_t = 0
    zsensor_t = 0
    zreal_t = 0

    for l_x, l_y, L_x, L_y, z_range, z_bearing, Z_range, Z_bearing in detected_x, detected_y, real_x, real_y, z_range_array, z_bearing_array, Z_range_array, Z_bearing_array:   
        q = (L_x - x_t)**2 + (L_y - y_t)**2
        if H_t == 0:
            H_t = np.matrix([-L_x-x_t/np.sqrt(q),-L_y-y_t/np.sqrt(q),0],[L_y-y_t/q,-L_x-x_t/q,-1])
            Q_t = np.matrix([sigma_l_d**2,0],[0,sigma_l_theta**2])
            zsensor_t = np.matrix([z_range],[z_bearing])
            zreal_t = np.matrix([Z_range],[Z_bearing])
        
        else: 
            H_t = np.vstack((H_t, np.matrix([-L_x-x_t/np.sqrt(q),-L_y-y_t/np.sqrt(q),0],[L_y-y_t/q,-L_x-x_t/q,-1])))
            Q_t = np.vstack((Q_t,np.matrix([sigma_l_d**2,0],[0,sigma_l_theta**2])))
            zsensor_t = np.vstack((zsensor_t,np.matrix([z_range],[z_bearing])))
            zreal_t = np.vstack((zreal_t,np.matrix([Z_range],[Z_bearing])))
    
    K_t = np.dot(np.dot(sigma_t,H_t.T),np.inv(np.dot(np.dot(H_t,sigma_t),H_t.T)+Q_t))
    INOVA = zsensor_t - zreal_t

    DeltaP = np.dot(K_t,INOVA)
    x_t = x_t + np.dot(K_t,INOVA)[0]
    y_t = y_t + np.dot(K_t,INOVA)[1]
    th_t = th_t + np.dot(K_t,INOVA)[2]

    sigma_x = np.sqrt(sigma_t[0,0])
    sigma_y = np.sqrt(sigma_t[1,1])
    sigma_xy = sigma_t[1,0]
    
    a = np.sqrt(1/2*(sigma_x**2+sigma_y**2+np.sqrt((sigma_y**2-sigma_x**2)**2+4*sigma_xy**2)))
    b = np.sqrt(1/2*(sigma_x**2+sigma_y**2-np.sqrt((sigma_y**2-sigma_x**2)**2+4*sigma_xy**2)))

    beta = 1/2*(np.arctan(2*sigma_xy/(sigma_y**2-sigma_x**2)))

    beta = normAngle(beta)

    kinematic_next =  np.array([(delta_r+delta_l)/2*np.cos(th_kinematic+(delta_r-delta_l)/(4*axis)),(delta_r+delta_l)/2*np.sin(th_kinematic+(delta_r-delta_l)/(4*axis)),(delta_r-delta_l)/(2*axis)])
    kinematic = kinematic + kinematic_next
    
    x_kinematic = kinematic[0]
    y_kinematic = kinematic[1]
    th_kinematic = kinematic[2]

    th_kinematic = normAngle(th_kinematic)
   
    X,Y = calculateEllipse(x_kinematic,y_kinematic,b,a,beta)
    ax.quiver(x_kinematic,y_kinematic,np.cos(th_kinematic),np.sin(th_kinematic),width=0.0005)
    ax.scatter(x_kinematic,y_kinematic,color='r')
    ax.plot(X,Y)

    x_odometry,y_odometry,th_odometry = getPose()
    ax.quiver(x_odometry,y_odometry,np.cos(th_odometry),np.sin(th_odometry),width=0.0005)
    ax.scatter(x_odometry,y_odometry,color='b')

    plt.pause(timespan)
    plt.draw()

    th_old = th_kinematic
    sigma_p_old = sigma_p



