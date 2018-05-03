#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
#import all necessary packages and utilities
import cv2
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Find Average Background Test: Takes in a single image and ")
    print("     averages the values of all pixels and returns ")
    print("     the average value in R,G,B scale. ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'dogball_closeup.png'
    fullFilePath = pathFwd + filename

    #read in image from camera (debug denotes thest image)
    joeCamera = imgFeedCamera(debug=fullFilePath)
    image = joeCamera.getFrame()

    #create the toolbox to be used for this test
    tools = VisionTools()

    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    final = tools.avgColor(image)
    print(final[0, 0])
    cv2.imshow('image', final)
    cv2.waitKey(0)



