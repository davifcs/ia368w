import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import restthru
from FeatureDetection_distances import FeatureDetection

host = 'http://127.0.0.1:4950'
restthru.http_init()

heuristic = True
Ks = 0.05
b_axis = 165
R_sigma_x = 0.5
R_sigma_y = 0.5
R_sigma_th = 0.3*np.pi/180
sigma_l_d = 0.5
sigma_l_theta = 0.1 * np.pi/180

timespan = 1

L_x = [0,  0,   920,  920,  4190,  4680, 4190,  4030, 4680]
L_y = [0,  2280, 2280, 3200, 2365,  2365, 3200,  0,    650]

L = np.array([L_x,L_y])

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
    
    res['th'] = normAngle(res["th"]*np.pi/180)

    return res

def getVel():
    vel2 = "/motion/vel2"
    res,_ = restthru.http_get(host+vel2)

    return res["left"], res["right"]
    
def getDistances():
    distances = "/perception/laser/1/distances"
    res,_ = restthru.http_get(host+distances)

    return res

def postPose(delta_pose):
    pose = "/motion/pose"
    restthru.http_post(host+pose,delta_pose)



PoseR = getPose()
theta_t_old = PoseR['th']
sigma_t_old = np.zeros([3,3])

R = np.matrix([[R_sigma_x,0,0],[0,R_sigma_y,0],[0,0,R_sigma_th]])


plt.ion()
fig, ax = plt.subplots()
plt.xlim(-1000,5500)
plt.ylim(-1000,4000)
count = 0

while(count < 60):
    pre_filter = datetime.now().timestamp()    
    #Prediction step
    l_vel,r_vel = getVel()

    delta_s_l = l_vel * timespan
    delta_s_r = r_vel * timespan

    delta_theta_t = (delta_s_r - delta_s_l)/(2*b_axis)
    delta_theta_t = normAngle(delta_theta_t)

    delta_s_t = (delta_s_r + delta_s_l)/2

    G_t = np.matrix([[1,0,-delta_s_t*np.sin(theta_t_old+delta_theta_t/2)],[0,1,delta_s_t*np.cos(theta_t_old+delta_theta_t/2)],[0,0,1]])

    a = 1/2*np.cos(theta_t_old+delta_theta_t/2)-delta_s_t/(4*b_axis)*np.sin(theta_t_old+delta_theta_t/2)
    b = 1/2*np.sin(theta_t_old+delta_theta_t/2)+delta_s_t/(4*b_axis)*np.cos(theta_t_old+delta_theta_t/2)
    c = 1/2*np.cos(theta_t_old+delta_theta_t/2)+delta_s_t/(4*b_axis)*np.sin(theta_t_old+delta_theta_t/2)
    d = 1/2*np.sin(theta_t_old+delta_theta_t/2)-delta_s_t/(4*b_axis)*np.cos(theta_t_old+delta_theta_t/2)

    V_t = np.matrix([[a,b],[c,d],[1/(2*b_axis),-1/(2*b_axis)]])

    sigma_delta_t = np.matrix([[Ks * np.abs(delta_s_r),0],[0,Ks * np.abs(delta_s_l)]])

    sigma_t_ = np.dot(np.dot(G_t,sigma_t_old),G_t.T) + np.dot(np.dot(V_t,sigma_delta_t),V_t.T) + R
        
    after_filter = datetime.now().timestamp()

    diff = after_filter - pre_filter
    
    # Update step
    time.sleep(timespan - diff)

    Distances = getDistances()
    PoseR = getPose()
    x_t = PoseR['x']
    y_t = PoseR['y']
    theta_t = PoseR['th']

    detected_x, detected_y, real_x, real_y, deltaBearing_array, z_range_array, z_bearing_array, Z_range_array, Z_bearing_array = FeatureDetection(Distances, L, PoseR)
    
    if not detected_x:
        continue

    H_t = np.array([])
        
    for (l_x, l_y, L_x, L_y, z_range, z_bearing, Z_range, Z_bearing) in zip(detected_x, detected_y, real_x, real_y, z_range_array, z_bearing_array, Z_range_array, Z_bearing_array):   
        q = (L_x - x_t)**2 + (L_y - y_t)**2
        if H_t.size == 0:
            H_t = np.matrix([[-(L_x-x_t)/np.sqrt(q),-(L_y-y_t)/np.sqrt(q),0],[(L_y-y_t)/q,-(L_x-x_t)/q,-1]])
            Q_t_array = np.array([sigma_l_d**2,sigma_l_theta**2])
            zsensor_t = np.matrix([[z_range],[z_bearing]])
            zreal_t = np.matrix([[Z_range],[Z_bearing]])
        
        else: 
            H_t = np.vstack((H_t, np.matrix([[-(L_x-x_t)/np.sqrt(q),-(L_y-y_t)/np.sqrt(q),0],[(L_y-y_t)/q,-(L_x-x_t)/q,-1]])))
            Q_t_array = np.hstack((Q_t_array,np.array([sigma_l_d**2,sigma_l_theta**2])))
            zsensor_t = np.vstack((zsensor_t,np.matrix([[z_range],[z_bearing]])))
            zreal_t = np.vstack((zreal_t,np.matrix([[Z_range],[Z_bearing]])))

    Q_t = np.eye(Q_t_array.size)
    row, col = np.diag_indices(Q_t_array.shape[0])
    Q_t[row,col] = Q_t_array

    K_t = np.dot(np.dot(sigma_t_,H_t.T),np.linalg.inv(np.dot(np.dot(H_t,sigma_t_),H_t.T)+Q_t))
    INOVA = zsensor_t - zreal_t

    DeltaP = np.dot(K_t,INOVA)

    PoseR = getPose()

    DeltaP = {
    "th": DeltaP[2].item(),
    "x": DeltaP[0].item(),
    "y": DeltaP[1].item()
    }
    
    x_t_no_filter = PoseR['x']
    y_t_no_filter = PoseR['y']
    th_t_no_filter = PoseR['th'] 

    if heuristic:
        mode = "heuristic"           
        if abs(delta_theta_t) < 0.01 and len(detected_x) > 1: 
            postPose(DeltaP)
            sigma_t = np.dot(np.eye(3) - np.dot(K_t,H_t),sigma_t_)
            sigma_t_old = sigma_t
            theta_t_old = theta_t
    else:
        mode = "no-heuristic"
        postPose(DeltaP)
        sigma_t = np.dot(np.eye(3) - np.dot(K_t,H_t),sigma_t_)
        sigma_t_old = sigma_t
        theta_t_old = theta_t        

        
    PoseR = getPose()

    x_t = PoseR['x']
    y_t = PoseR['y']
    th_t = PoseR['th'] 

    ax.quiver(x_t_no_filter,y_t_no_filter,np.cos(th_t_no_filter),np.sin(th_t_no_filter),width=0.00005)
    ax.scatter(x_t_no_filter,y_t_no_filter,color='r')

    ax.quiver(x_t,y_t,np.cos(th_t),np.sin(th_t),width=0.00005)
    ax.scatter(x_t,y_t,color='b')

    plt.pause(0.0001)
    plt.draw()

    count += 1
fig.savefig(mode + ".png") 


    


