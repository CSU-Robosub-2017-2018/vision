#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
#import all necessary packages and utilities
import cv2
import context
import numpy as np
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Find Object With Color Test: Takes in a single image and ")
    print(" determines if there is an object based on a user defined ")
    print(" color. Returns a masked image with the object as white ")
    print("     and everything else as black. (Uses BGR scale) ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #set the color the user wishes to find
    userColor = "red"

    #read in image from camera (debug denotes test image)
    joeCamera = imgFeedCamera(debug="/home/oren/vision/tests/test_files/circles.png")
    image = joeCamera.getFrame()
    
    #show original image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
    tools = VisionTools()

    mask = tools.findObjWColor(image, userColor)

    #show color masked image
    cv2.imshow('mask',mask)
    cv2.waitKey(0)
