#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
# import all necessary packages and utilities
import cv2
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera


if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Buoy Bounding Boxes Test: Takes in a single image and ")
    print("     draws bounding boxes around discovered 'buoys'.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # read in image from camera (debug denotes test image)
    #define relative path to test file
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'Circle.png'
    fullFilePath = pathFwd + filename

    joeCamera = imgFeedCamera(debug=fullFilePath)
    image = joeCamera.getFrame()

    # Resize Image
    r = 640.0 / image.shape[1]
    dim = (640, int(image.shape[0] * r))

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    tools = VisionTools()

    final = tools.BuoyBoxes(image)
    cv2.imshow('image', final)
    cv2.waitKey(0)
