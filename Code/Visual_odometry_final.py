from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import numpy as np
import random
from numpy.linalg import matrix_rank
import math
import cv2
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd

# Computing Calibration Matrix K 
fx, fy, c_x, c_y, camera_img, LUT = ReadCameraModel('model/') 
K = np.array([[fx , 0 , c_x],[0 , fy , c_y],[0 , 0 , 1]])


def get_images():
    path = "stereo/centre/"
    frames = []
    for frame in os.listdir(path):
        frames.append(frame)
    frames.sort()
    return frames


def data_prep(img):
    # Converting Bayers to BGR
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    # Undistorting the image
    undistorted_img = UndistortImage(BGR_img,LUT)  
    # Converting to grayscale
    gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    # Masking the Car region from the frames
    ROI = gray_img[200:650, 0:1280]
    return gray_img


def SIFT_matches(frame1, frame2):
    features1 = []
    features2 = []
    
    sift = cv2.xfeatures2d.SIFT_create()        #creating an object of SIFT module
    FLANN_INDEX_KDTREE = 0      #algorithm label
    
    #creating dictionaries to store index params and search params to be used in FlannBasedMatcher function
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    # Get the keypoints and matrices from 2 frames
    kp1, des1 = sift.detectAndCompute(frame1,None)
    kp2, des2 = sift.detectAndCompute(frame2,None)
    
    # Find the point correspondence between the two frames
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)        #identifies a set of common keypoints between des1 and des2     

    # Considering only matching points that are sufficiently spaced away from each other
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            #generates a set of common features for respective frame coordinates
            features1.append(kp1[m.queryIdx].pt)        
            features2.append(kp2[m.trainIdx].pt)
    
    return features1, features2

#Function to identify eight random points with similar features for both images
def get_random_eight_points(features1, features2):
    feature1_list = []
    feature2_list = []
    random_points = []
    while(True):
        num = random.randint(0, len(features1)-1)       #choosing a random index point in features1 list
        if num not in random_points:        #if index number is not already in list
            random_points.append(num)       
        if len(random_points) == 8:     #break when we have 8 points
            break

    for point in random_points:
        feature1_list.append([features1[point][0], features1[point][1]])
        feature2_list.append([features2[point][0], features2[point][1]])
    return feature1_list, feature2_list

def Camera_pose(E_matrix):
    u, s, v = np.linalg.svd(E_matrix, full_matrices=True)   #doing SVD for New E matrix (section 3.4)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])    #drafting w matrix    
    
    t1 = u[:, 2]        #C1 = third column
    r1 = u @ w @ v      #R1 = UWV'
    t2 = -u[:, 2]       #C2 = - third column
    r2 = u @ w @ v      #R2 = UWV'
    t3 = u[:, 2]        #C3 = third column
    r3 = u @ w.T @ v    #R3 = UW'V'  
    t4 = -u[:, 2]       #C4 = -third column
    r4 = u @ w.T @ v    #R4 = UW'V'
    
#Further checks to check if det is negative
    if np.linalg.det(r1) < 0:
        t1 = -t1 
        r1 = -r1

    if np.linalg.det(r2) < 0:
        t2 = -t2 
        r2 = -r2 

    if np.linalg.det(r3) < 0:
        t3 = -t3 
        r3 = -r3 

    if np.linalg.det(r4) < 0:
        t4 = -t4 
        r4 = -r4 
        
    t2 = t2.reshape((3,1))
    t1 = t1.reshape((3,1))
    t3 = t3.reshape((3,1)) 
    t4 = t4.reshape((3,1))
    
    return [r1, r2, r3, r4], [t1, t2, t3, t4]

def obtainEulerAngles(rot_mat) :
    eu1 = math.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
    singular_val = eu1 < 1e-6       #boolean to check if value is less than 1e-6
    if  not singular_val :
        x = math.atan2(rot_mat[2,1] , rot_mat[2,2])
        y = math.atan2(-rot_mat[2,0], eu1)
        z = math.atan2(rot_mat[1,0], rot_mat[0,0])

    else :
        x = math.atan2(-rot_mat[1,2], rot_mat[1,1])
        y = math.atan2(-rot_mat[2,0], eu1)
        z = 0
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])


def Fundamental_Matrix(frame1_features, frame2_features): 
    A_x = np.empty((8, 9))      #creating a 8x9 matrix for SVD calculation

    for i in range(0, len(frame1_features)):
        x_1 = frame1_features[i][0]
        y_1 = frame1_features[i][1]
        x_2 = frame2_features[i][0]
        y_2 = frame2_features[i][1]
        
        #modelled as going from frame 2 to frame 1 here
        #Each row modelled as [x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]
        A_x[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])
        # A_x[i] = np.array([x_2*x_1, x_1*y_2, x_1, y_1*x_2, y_1*y_2, y_1, x_2, y_2, 1])
        
    # SVD to calculate F matrix
    u, s, v = np.linalg.svd(A_x, full_matrices=True)  
    f = v[-1].reshape(3,3)      #taking the final row as per our requirement from SVD and creating new F
    # But this value of F might not actually have rank 3. So, we force the constraint here.
    
    # Forcing rank constraint of 2
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) #making the last element manullay to zero (s2[3,3]=0) -> rank 2
    
    Fundamental_matrix = u1 @ s2 @ v1       #resultant fundamental matrix has rank 2
    return Fundamental_matrix

def condition_check(x1, x2, F):         #matrix to check for fundamental matrix validation
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11))) #squeezing to get rid of extra rows

def Essentialmatrix(K, Fmatrix):        #Creating Essential Matrix from F matrix
    E = np.matmul(np.matmul(K.T, Fmatrix), K)
    u, s, v = np.linalg.svd(E, full_matrices=True)  #using SVD and finding V, which is our requirement
    
    #imposing rank 2 criteria check for essential matrix
    S_F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])       #rank -> 2
    temp = np.matmul(u, S_F)
    E_matrix = np.matmul(temp, v)               #Finding out resultant E Matrix
    return E_matrix

def H_matrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

def getTriangulationPoint(m1, m2, point1, point2):
    oldx = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]])
    oldxdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    A1 = oldx @ m1[0:3, :] 
    A2 = oldxdash @ m2
    A_x = np.vstack((A1, A2))
    u, s, v = np.linalg.svd(A_x)
    new1X = v[-1]
    new1X = new1X/new1X[3]
    new1X = new1X.reshape((4,1))
    return new1X[0:3].reshape((3,1))

def Final_camera_pose(R, C, features1, features2):
    check = 0
    Horigin = np.identity(4)
    for index in range(0, len(R)):
        angles = obtainEulerAngles(R[index])        #finds euler angles x,y,z
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:   #we are concerned only about angles in the XZ plane (thats our plane of focus)
            count = 0
            newP = np.hstack((R[index], C[index]))
            for i in range(0, len(features1)):
                temp1x = getTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i])
                thirdrow = R[index][2,:].reshape((1,3))
                if np.squeeze(thirdrow @ (temp1x - C[index])) > 0: 
                    count = count + 1
            if count > check:
                check = count
                Translation_final = C[index]
                Rotation_final = R[index]
                
    if Translation_final[2] > 0:
        Translation_final = -Translation_final
                
    return Rotation_final, Translation_final

def get_tempfeatures(features1, features2, F_Matrix, count):
    #listing features that are compliant with the chosen fundamental matrix only
    TemporaryFeatures_1 = []
    TemporaryFeatures_2 = []
    for number in range(0, len(features1)):
        if condition_check(features1[number], features2[number], F_Matrix) < 0.01:
            count = count + 1
            TemporaryFeatures_1.append(features1[number])
            TemporaryFeatures_2.append(features2[number])
    return TemporaryFeatures_1, TemporaryFeatures_2, count


def image_process(H_init,p_0, all_images):
    data_points = []
    
    for index in range(20, len(all_images)-1):
    # for index in range(20, 100):
        Inliers = 0             #setting inliers to zero for each iteration
        print("image_number = ",index)
        
    
        
        img1 = cv2.imread("stereo/centre/" + str(all_images[index]), 0)         #loading first image in greyscale
        img2 = cv2.imread("stereo/centre/" + str(all_images[index + 1]), 0)     #loading second image in greyscale
        
        colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)     #converting first image from BayerGR to BGR
        colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)     #converting second image from BayerGR to BGR
        
        undistortedimage1 = UndistortImage(colorimage1,LUT)     #undistorting first image using LUT
        undistortedimage2 = UndistortImage(colorimage2,LUT)     #undistorting second image using LUT
        
        gray1 = cv2.cvtColor(undistortedimage1,cv2.COLOR_BGR2GRAY)      #converting back to grayscale
        gray2 = cv2.cvtColor(undistortedimage2,cv2.COLOR_BGR2GRAY)      #converting back to grayscale
        
        grayImage1 = gray1[200:650, 0:1280]     #creating a mask for region without car
        grayImage2 = gray2[200:650, 0:1280]     #creating a mask for region without car
        
        #  Finding proper matches between two frames
        features1, features2 = SIFT_matches(grayImage1, grayImage2)

        # RANSAC to find the best F matrix (running for 45 iterations)
        for i in range(0, 45):
            count = 0
            #generate 2 feature lists for 8 random matching points in both frames
            Frame1_features, Frame2_features=  get_random_eight_points(features1, features2)
      
            #create Fundamental matrix of rank 2 for 2 frames
            F_Matrix = Fundamental_Matrix(Frame1_features, Frame2_features)
            
            #finding temp features from the features list that completely match with the fundamental matrix
            TemporaryFeatures_1, TemporaryFeatures_2, count = get_tempfeatures(features1, features2, F_Matrix, count)
            if count > Inliers:
                Inliers = count
                inlier1 = TemporaryFeatures_1
                inlier2 = TemporaryFeatures_2
                FinalFundamentalMatrix = F_Matrix
        
        # Calculating the Essential matrix
        E_matrix = Essentialmatrix(K, FinalFundamentalMatrix)
        
        # Estimating Camera pose from Essential matrix and obtaining Rot and Trans matrices
        RotationMatrix, Tlist = Camera_pose(E_matrix)
        
        # Triangulating to get the final Rotation and Translation matrix which has positive Z depth 
        rot_mat, T = Final_camera_pose(RotationMatrix, Tlist, inlier1, inlier2)

        H_init = H_init @ H_matrix(rot_mat, T)  #matmul between H_init(Ident 4x4) and H_matrix that was calculated using new Rot and Trans matrix 
        p_projection = H_init @ p_0     #new projection point calculated

        data_points.append([p_projection[0][0], -p_projection[2][0]])
        
        plt.scatter(p_projection[0][0], -p_projection[2][0], color='r')
    return data_points


def write_to_file(data_points):
    File_1=open("plotpoints.txt", "w+")
    for data in data_points:
        File_1.write(str(data[0]))
        File_1.write(',')
        File_1.write(str(data[1]))
        File_1.write('\n')
    File_1.close()

def main():
    all_images = get_images()
    H_init = np.identity(4)
    p_0 = np.array([[0, 0, 0, 1]]).T
    data = image_process(H_init,p_0, all_images)
    print(data)
    write_to_file(data)
    plt.show()

    

if __name__ == '__main__':
    main()