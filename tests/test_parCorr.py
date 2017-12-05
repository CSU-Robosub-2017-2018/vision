#!/usr/bin/env python
'''
test_simple.py - Determines center, size, associated mask => location in image
                 Utilizes a dynamic environment to drastically reduce computation time for same result

                 '''

# import all necessary packages and utilities
from pprint import pprint
import cv2
import context
from vision.vision_tools import VisionTools
from vision.cameras.camera_img_feed import imgFeedCamera


if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Shape Correlation Test:  Determines center, size, ") 
    print("     associated mask => location in image ")
    print("     utilizes a dynamic environment to drastically reduc ")
    print("     computation time for same result ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    match_type = cv2.TM_CCORR_NORMED
    #user inputs go here:
    #div = num of masks
    #userMask = shape to find
    #size_range = max scaling of the mask

    userMask = "circle"
    div = 20
    size_range = 200

    #load in environment image in grayscale
    #environment = cv2.imread('/home/oren/vision/tests/test_files/env4.png', cv2.IMREAD_GRAYSCALE)
    joeCamera = imgFeedCamera(debug="/home/oren/vision/tests/test_files/env4.png")
    environment = joeCamera.getFrame()
    environment = cv2.cvtColor(environment, cv2.COLOR_RGB2GRAY) 
    #cv2.imshow('environment', environment)
    #cv2.waitKey(0)

    tools = VisionTools()

    locx, locy, max_val, final_obj = tools.calcObject(environment, userMask, div, size_range)

    pprint(locx)
    pprint(locy)
    pprint(max_val)
    pprint(final_obj)


