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
from vision.cameras.camera_video_feed import videoFeedCamera


if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Line Detection Test: Takes in a video and runs a line detection")
    print("      filtering algorithm on it then displays frame by frame.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'output.avi'
    fullFilePath = pathFwd + filename

    joeCamera = videoFeedCamera(debug=fullFilePath)
    tools = VisionTools()

    # Setup a loop to capture a frame from the video feed
    while True:
        image = joeCamera.getFrame()
        final = tools.lineDet(image)
        cv2.imshow("Filtered line", final)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
