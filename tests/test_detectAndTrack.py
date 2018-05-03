#!/usr/bin/env python
'''
test_simple.py - test that combines the functionality of the detaction
and tracking tests. Processing is done on a video
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
    print("Object Detect and Track Test: Combination of detection ")
    print("and tracking tests. Detects an object as soon as it ")
    print("enters the environment, draws a box around it, and")
    print("tracks it until it leaves the screen. This works for ")
    print("orange, green, and blue objects.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    match_type = cv2.TM_CCORR_NORMED
    #tracker starts out not initialized
    track_ok = False
    frame_count = 0
    #define shape to be found. square and triangle must be added later. See comments in detectObject function in vision_tools
    userMasks = ['circle', 'square', 'triangle']
    userMask = userMasks[0]
    #define the desired color of the object
    userColors = ['orange', 'green', 'red']
    userColor = userColors[0]
    #define detection tolerance: likelyhood that detected object fits userMask (ex. 0.95 -> 95% match)
    threshold = .4
    #different trackers that can be used
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    #each tracker has its own benefits and downfalls. Visit learnopencv.com for details
    #for all intents and purposes now, we will be using a MIL tracker
    tracker_type = tracker_types[2]
    #define relative path to test file
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'clip2_GOPR0157.MP4'
    fullFilePath = pathFwd + filename

    # instantiate video camera
    joeCamera = videoFeedCamera(debug=fullFilePath)
    tools = VisionTools()

    while True:
        #read new frame and confirm read
        success, environment = joeCamera.getFrame()
        if not success:
            break
        #preprocess environment
        environment = cv2.resize(environment, (1019, 589))
        blur = cv2.GaussianBlur(environment,(5,5),0)
        mask_env = tools.findObjWColor(blur, userColor)
        #convert to gray scale for detection algorithm. NOT CURRENTLY USED
        #gray_environment = cv2.cvtColor(environment, cv2.COLOR_RGB2GRAY)

        #run detect on entire environment for the first frame and every 30 frames after that
        #TODO if something is in the frame and big_ROI is created, need to run detect on entire environment EXCEPT FOR big_ROI. Read further to see where big_ROI comes from
        if frame_count == 0 or frame_count % 30 == 0:
            obj_cent, locx, locy, max_val, final_obj, small_ROI, detect = tools.detectObject(environment, mask_env, userMask, threshold)
            if detect == True:
                #check if the avg color of the small ROI falls in the range of a specified color
                color = tools.avgColor(small_ROI)
                RGB = color[0, 0]
                color_check = tools.checkColor(RGB)
                if color_check == True:
                    #draw a bounding box around the object, define the bounding box, calculate the percentage of the
                    #frame covered by the bounding box, and define a region of interest (ROI) around the boxed object
                    p1, p2, bbox, percent_area, big_ROI, mask_big_ROI = tools.BBoxAndROIS(obj_cent, environment, userColor)
                    #boxed_obj = cv2.rectangle(environment, p1, p2, (255,0,0), 2, 1)
                    #initialize tracker
                    track_ok, tracker = tools.trackerInit(environment, tracker_type, bbox)

                else:
                    frame_count += 1
                    if frame_count == 1000:
                        frame_count = 0
                    continue

        if track_ok and frame_count % 10 == 0:
            #re-detect object in big_ROI
            obj_cent, locx, locy, max_val, final_obj, small_ROI, detect = tools.detectObject(big_ROI, mask_big_ROI, userMask, threshold)
            p1, p2, bbox, percent_area, big_ROI, gray_big_ROI = tools.BBoxAndROIS(obj_cent, big_ROI, userColor)

        if track_ok:
            #track object bounded by bounding box
            tracking = tools.track(environment, tracker_type, bbox, tracker, track_ok)

            # Display result
            cv2.imshow("Tracking", tracking)
            frame_count += 1
            if frame_count == 1000:
                frame_count = 0
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow("Tracking", environment)
            frame_count += 1
            if frame_count == 1000:
                frame_count = 0
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            else:
                continue


