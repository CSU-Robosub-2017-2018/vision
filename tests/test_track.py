#!/usr/bin/env python
'''
test_simple.py - this is an example taken from learnopencv.com that was modified
for the purpose of the RoboSub Project
'''

#import all necessary packages and utilities
import cv2
from pprint import pprint
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_video_feed import videoFeedCamera

if __name__ == '__main__':

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Object Tracker Test: uses various types of trackers ")
    print("to track an object defined by a bounding box created by ")
    print("the user. The video is taken in by means of the camera ")
    print("object. Frame rate and Tracker Type display on screen. ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'clip_GOPR0157.MP4'
    fullFilePath = pathFwd + filename
    # instantiate video camera
    # FIXME video images need to be resized. run test and you will see why
    joeCamera = videoFeedCamera(debug=fullFilePath)

    tools = VisionTools()

    #read first frame and confirm that frame was actually read
    success, frame = joeCamera.getFrame()

    #select a user defined bounding box on the initial frame
    bbox = cv2.selectROI(frame, False)

    #different trackers that can be used
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    #each tracker has its own benefits and downfalls. Visit learnopencv.com for details
    #for all intents and purposes now, we will be using a MIL tracker
    tracker_type = tracker_types[2]

    #initialize tracker
    ok, tracker = tools.trackerInit(frame, tracker_type, bbox)

    while True:
        #read new frame and confirm read
        success, frame = joeCamera.getFrame()
        if not success:
            break

        #track object bounded by bounding box
        final = tools.track(frame, tracker_type, bbox, tracker, ok)

        # Display result
        cv2.imshow("Tracking", final)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

