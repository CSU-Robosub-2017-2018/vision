#!/usr/bin/env python
'''
test_simple.py - Determines center, size, associated mask => location in image
                 Utilizes a dynamic environment to drastically reduce computation time for same result

                 '''

# import all necessary packages and utilities
from pprint import pprint
import cv2
import context
import os
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera


if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Object Detection Test:  Determines center, size, ")
    print("     associated mask => location in image ")
    print("     utilizes a dynamic environment to drastically reduce ")
    print("     computation time for same result ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    match_type = cv2.TM_CCORR_NORMED
    #define shape to be found
    userMask = "circle"
    userColor = "orange"
    #define detection tolerance: likelyhood that detected object fits userMask (ex. 0.95 -> 95% match)
    threshold = 0.4
    #define relative path of test files
    path = os.getcwd()
    #in case of windows users, switched backslashes with fwd slashes
    pathFwd = '/'.join(path.split('\\'))
    pathFwd = pathFwd + '/test_files/'
    filename = 'oball_screenshot.png'
    fullFilePath = pathFwd + filename
    #print("The current working directory is", path)
    #load in environment image
    joeCamera = imgFeedCamera(debug=fullFilePath)
    tools = VisionTools()
    environment = joeCamera.getFrame()


    #preprocess environment
    blur = cv2.GaussianBlur(environment,(5,5),0)
    mask_env = tools.findObjWColor(blur, userColor)
    #gray_mask_env = cv2.cvtColor(mask_env, cv2.COLOR_RGB2GRAY)
    cv2.imshow('env', environment)
    #cv2.imshow('blur', blur)
    #cv2.imshow('mask_env', mask_env)
    #cv2.imshow('gray_mask_env', gray_mask_env)
    cv2.waitKey(0)


    #detect object based on mask defined by userMask
    obj_cent, locx, locy, max_val, final_obj, small_ROI, detect = tools.detectObject(environment, mask_env, userMask, threshold)
    color = tools.avgColor(small_ROI)
    rgb = color[0, 0]

    #draw a bounding box around the object, define the bounding box, calculate the percentage of the
    #frame covered by the bounding box, and define a region of interest (ROI) around the boxed object
    p1, p2, bbox, percent_area, big_ROI, mask_big_ROI = tools.BBoxAndROIS(obj_cent, environment, userColor)
    boxed_obj = cv2.rectangle(environment, p1, p2, (255,0,0), 2, 1)
    cv2.imshow('boxed object', boxed_obj)
    #cv2.imshow('avg', color)

    #cv2.imshow('big_ROI', big_ROI)
    #cv2.imshow('small_ROI', small_ROI)

    pprint(obj_cent)
    pprint(bbox)
    pprint(percent_area)
    cv2.waitKey(0)
