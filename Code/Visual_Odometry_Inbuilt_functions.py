import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

fx, fy, c_x, c_y, camera_img, LUT = ReadCameraModel('model/') 
K = np.array([[fx , 0 , c_x],[0 , fy , c_y],[0 , 0 , 1]])

def SIFT_matches(frame1, frame2):
    features1 = []
    features2 = []
    
    sift = cv2.xfeatures2d.SIFT_create()        #creating an object of SIFT module
    
    #creating dictionaries to store index params and search params to be used in FlannBasedMatcher function
    FLANN_INDEX_KDTREE = 0
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
        if m.distance < 1*n.distance:
            #generates a set of common features for respective frame coordinates
            features1.append(kp1[m.queryIdx].pt)        
            features2.append(kp2[m.trainIdx].pt)
    features1 = np.int32(features1)
    features2 = np.int32(features2)
    return features1, features2    
    
    
def get_images():
    path = "stereo/centre/"
    frames = []
    for frame in os.listdir(path):
        frames.append(frame)
    frames.sort()
    return frames

def H_matrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

def image_process(H_init,p_0, all_images):
    data_points = []
    
    for index in range(20, len(all_images)-1):
        print("image_number = ",index)
        
        img1 = cv2.imread("stereo/centre/" + str(all_images[index]), 0)         #loading first image in greyscale
        img2 = cv2.imread("stereo/centre/" + str(all_images[index + 1]), 0)     #loading second image in greyscale
        
        colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)     #converting first image from BayerGR to BGR
        colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)     #converting second image from BayerGR to BGR
        
        undistortedimage1 = UndistortImage(colorimage1,LUT)     #undistorting first image using LUT
        undistortedimage2 = UndistortImage(colorimage2,LUT)     #undistorting second image using LUT
        
        gray1 = cv2.cvtColor(undistortedimage1,cv2.COLOR_BGR2GRAY)      #converting back to grayscale
        gray2 = cv2.cvtColor(undistortedimage2,cv2.COLOR_BGR2GRAY)      #converting back to grayscale
        
        #  Finding proper matches between two frames
        features1, features2 = SIFT_matches(gray1, gray2)
        
        #using inbuilt function to directly compute F_matrix after RANSAC
        FinalFundamentalMatrix, m = cv2.findFundamentalMat(features1, features2, cv2.FM_RANSAC)
        features1 = features1[m.ravel() == 1]
        features2 = features2[m.ravel() == 1]
        
        
        # Calculating the Essential matrix
        E_matrix = K.T @ FinalFundamentalMatrix @ K
        
        # Obtaining final Rotational and Translation pose for cameras
        retval, rot_mat, T, mask = cv2.recoverPose(E_matrix, features1, features2, K)
        
        H_init = H_init @ H_matrix(rot_mat, T)  #matmul between H_init(Ident 4x4) and H_matrix that was calculated using new Rot and Trans matrix 
        p_projection = H_init @ p_0     #new projection point calculated

        data_points.append([p_projection[0][0], -p_projection[2][0]])
        
        plt.scatter(p_projection[0][0], -p_projection[2][0], color='g')
    return data_points

def write_to_file(data_points):
    File_1=open("plotpoints_inbuilt.txt", "w+")
    for data in data_points:
        File_1.write(str(data[0]))
        File_1.write(',')
        File_1.write(str(data[1]))
        File_1.write('\n')
    File_1.close()
    
def main():
    all_images=get_images()  
    H_init = np.identity(4)
    p_0 = np.array([[0, 0, 0, 1]]).T
    data = image_process(H_init,p_0, all_images)
    # print(data)
    write_to_file(data)
    plt.show()

if __name__ == '__main__':
    main()