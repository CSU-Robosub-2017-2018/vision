#!/usr/bin/env python
'''
test_simple.py - Tests registration of a single video against a background
                 image, detection and classification of things, and
                 identification of the classified things on screen.
                 '''
import cv2
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_video_feed import videoFeedCamera

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("     Buoy Bounding Boxes Test: Takes in a video and         ")
    print("     draws bounding boxes around discovered 'buoys'.        ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Read original video, alter for different videos
    #define relative path to test file
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'Test1.mp4'
    fullFilePath = pathFwd + filename

    joeCamera = videoFeedCamera(debug=fullFilePath)

    tools = VisionTools()


    while(True):
        image = joeCamera.getFrame()

        cv2.imshow('image', image)

        # Resize Image
        r = 640 / image.shape[1]
        dim = (640, int(image.shape[0] * r))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        final = tools.BuoyBoxes(image)


        cv2.imshow('image', final)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
