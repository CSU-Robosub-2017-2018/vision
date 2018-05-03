#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
#import all necessary packages and utilities
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Scan ROI test: recieves an image as an imput, creates a ")
    print("region of interest (ROI), and scans that ROI across the")
    print(" entire image. Press 'q' to stop scan. ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'circles.png'
    fullFilePath = pathFwd + filename
    #read in image from camera (debug denotes thest image)
    joeCamera = imgFeedCamera(debug="/home/oren/vision/tests/test_files/circles.png")
    image = joeCamera.getFrame()

    #create the toolbox to be used for this test
    tools = VisionTools()

    final = tools.roiScan(image, 100)

