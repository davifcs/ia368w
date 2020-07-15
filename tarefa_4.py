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
range_min = -90
range_max = 90
range_step = 1
laser_range = [range_min,range_max,range_step]
mark_threshold = 30

timespan = 1

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
    distances = "/perception/laser/1/distances?range="+str(range_min)+":"+str(range_max)+":"+str(range_step)+""
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

X_t = np.empty([3])

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
    
    G_t_offset = sigma_t_old.shape[0]-G_t.shape[0]
    G_t = np.pad(G_t, ((0,G_t_offset), (0,G_t_offset)), mode='constant')
    R = np.pad(R, ((0,G_t_offset), (0,G_t_offset)), mode='constant')

    for i in range (G_t_offset, sigma_t_old.shape[0]):
        G_t[i][i] = 1

    sigma_t_ = np.dot(np.dot(G_t,sigma_t_old),G_t.T)# + np.dot(np.dot(V_t,sigma_delta_t),V_t.T) + R
        
    after_filter = datetime.now().timestamp()

    diff = after_filter - pre_filter
    
    # Update step
    time.sleep(timespan - diff)

    Distances = getDistances()
    PoseR = getPose()
    X_t[0] = PoseR['x']
    X_t[1] = PoseR['y']
    X_t[2] = PoseR['th']
    Features = FeatureDetection(Distances,laser_range)
    
    n_features = len(Features)

    if n_features == 0:
        continue
    
    for r,b in Features:
        k = None
        M_x = PoseR['x'] + r * np.cos(b + PoseR['th'])
        M_y = PoseR['y'] + r + np.sin(b + PoseR['th'])
        if X_t.size > 3:
            M_x_dist = abs(X_t[3::2] - M_x)
            M_y_dist = abs(X_t[4::2] - M_y)
            for x, y in zip(M_x_dist, M_y_dist):
                if x < mark_threshold and y < mark_threshold:
                    k, = np.where(M_x_dist == x)
                    k = k[0] + 3  
        if k is None:
            print(X_t)
            X_t = np.append(X_t,[M_x,M_y])
            sigma_t_ = np.pad(sigma_t_, ((0,2), (0,2)), mode='constant')
            sigma_len = sigma_t_.shape[0]
            sigma_t_[sigma_len-1][sigma_len-1] = sigma_l_d
            sigma_t_[sigma_len-2][sigma_len-2] = sigma_l_d
            continue

        H_t = np.array([])

        delta_x = M_x - X_t[0]    
        delta_y = M_y - X_t[1]
        delta = np.append(delta_x,delta_y)

        q = np.dot(delta.T,delta)

        F = np.zeros((5,X_t.shape[0]))
        F[0][0] = 1
        F[1][1] = 1
        F[2][2] = 1
        F[k][3] = 1
        F[k+1][4] = 1

        H_t = np.matrix([[-np.sqrt(q)*delta_x,-np.sqrt(q)*delta_y,0,np.sqrt(q)*delta_x,np.sqrt(q)*delta_y],[delta_y,-delta_x,-q,-delta_y,delta_x]])
        H_t = np.dot(H_t,F)
        Q_t = np.matrix([[0,sigma_l_d**2],[sigma_l_theta**2,0]])

        zsensor_t = np.matrix([[np.sqrt(q)],[np.arctan2(delta_y,delta_x)-X_t[2]]])
        zreal_t = np.matrix([[r],[b]])

        K_t = np.dot(np.dot(sigma_t_,H_t.T),np.linalg.inv(np.dot(np.dot(H_t,sigma_t_),H_t.T)+Q_t))
        INOVA = zsensor_t - zreal_t

        X_t = X_t.T + np.dot(K_t,INOVA)
        
        X_t = X_t[0]
        
        
        sigma_t_ = np.dot(np.eye(sigma_t_.shape[0]) - np.dot(K_t,H_t),sigma_t_)
    
    PoseR = np.array([PoseR['x'], PoseR['y'], PoseR['th']])
    PoseK = np.array([X_t[0], X_t[1], X_t[2]])
    DeltaP = PoseK - PoseR
    DeltaP[2] = normAngle(DeltaP[2])

    DeltaP = {
    "th": DeltaP[2].item(),
    "x": DeltaP[0].item(),
    "y": DeltaP[1].item()
    }
    
    postPose(DeltaP)
    
    sigma_t_old = sigma_t_

    plt.pause(0.0001)
    plt.draw()

    count += 1
fig.savefig("EKFSlam.png") 


    


