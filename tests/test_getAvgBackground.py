#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
#import all necessary packages and utilities
import cv2
import context
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Find Average Background Test: Takes in a single image and ")
    print("     averages the values of all pixels and returns ")
    print("     the average value in R,G,B scale. ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #read in image from camera (debug denotes thest image)
    joeCamera = imgFeedCamera(debug="/home/oren/vision/tests/test_files/background2.jpg")
    image = joeCamera.getFrame()

    #create the toolbox to be used for this test
    tools = VisionTools()

    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    final = tools.avgColor(image)

    cv2.imshow('image', final)
    cv2.waitKey(0)
    


