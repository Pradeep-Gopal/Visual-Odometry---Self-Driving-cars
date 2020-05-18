#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import cv2

def get_images():
    path = "stereo/centre/"
    frames = []
    for frame in os.listdir(path):
        frames.append(frame)
    frames.sort()
    return frames

def read_file():
    file = open('plotpoints.txt', "r")
    # file = open('plotpoints_inbuilt.txt', "r")
    c = []
    lines = file.readlines()
    for i, line in enumerate(lines):
        a=float(line.split(',')[0])
        b=float(line.split(',')[-1])
        c.append((a,b))
    return c

def main():
    all_images=get_images()
    cl = read_file()
    for i,c in enumerate(cl):
        img1 = cv2.imread("stereo/centre/" + str(all_images[i+20]), 0)
        cv2.imshow('img1', img1)
        cv2.waitKey(1)
        plt.scatter(c[0],c[1],color='r')
        plt.pause(0.0001)
    plt.show()
   
if __name__ == '__main__':
    main()
