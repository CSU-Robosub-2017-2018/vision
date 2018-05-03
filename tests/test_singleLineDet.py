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
    print("Line Detection Test: Takes in a single image and runs a line")
    print("     detection filtering algorithm on it then displays.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'line1.jpg'
    fullFilePath = pathFwd + filename

    joeCamera = imgFeedCamera(debug=fullFilePath)
    image = joeCamera.getFrame()
    tools = VisionTools()

    # Setup a loop to capture a frame from the video feed
    while True:
        final = joeCamera.getFrame()
        final = tools.lineDet(final)
        cv2.imshow("Filtered line", final)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
