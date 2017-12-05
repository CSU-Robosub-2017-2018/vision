'''
vision_tools.py - Tools for the vision portion of the RoboSub project.
Contains useful filters and processing routines for detection and
target tracking.
'''
import numpy as np
import cv2
import math


class VisionTools:
    ##
    # @brief Does nothing important for the time being.  Will add some
    #        functionality later


    def __init__(self):
        # Dummy variable, not used for anything
        self.useme = True

    ##
    # @brief finds object location based on shape: Developed by Jake Harmon
    # @param frame_env Image passed in from camera
    # @param userMask The shape the user/controls wishes to find
    # @param div number of slices in the size of the mask
    # @param size_range maximum scaling of the mask based on the original size
    # @return lock[x] -vector- X values of the location
    # @return lock[y] -vector- Y values of the location
    # @return finalObject Image of the object
    # @return max_val -vector- maximum correlation values for each mask

    def calcObject(self, frame_env, userMask, div, size_range):
        match_type = cv2.TM_CCORR_NORMED
        if userMask == "circle":
            obj = cv2.imread('/home/oren/vision/tests/test_files/circ_obj.PNG', cv2.IMREAD_GRAYSCALE)
        #if userMask == "square":
            #obj = cv2.imread('/home/oren/vision/tests/test_files/sq_obj.PNG'), cv2.IMREAD_GRAYSCALE)
        #if userMask == "triangle":
            #obj = cv2.imread('/home/oren/vision/tests/test_files/tri_obj.PNG'), cv2.IMREAD_GRAYSCALE)
        #imk stores the sliced stages of the mask
        imk = []

        for J in range(1, div):
            imk.append(cv2.resize(obj, (0,0), fx=J*div/size_range, fy=J*div/size_range))
            #size of mask from 10% of input size to 200% of input size
            #this range has been decided arbitrarily
        
        length = len(imk)
        env = [0]*length
        maxCorr = [0]*length
        maxTuple = [0]*length
        y = [0]*length
        x = [0]*length
        b= [0]*length
        c=[0]*length
        y1 = [0]*length
        y2 = [0]*length
        x1 = [0]*length
        x2 = [0]*length
        mn=[0]*length
        new_frame_env = [0]*length
        corrnew = [0]*length
        envir = [0]*length
        maxCorr_new = [0]*length
        maxTuple_new = [0]*length
        max_loc = [0]*length
        max_loc_new=[0]*length
        max_val = [0]*length
        max_val_new = [0]*length
        max_val_new = [0]*length
        intem_y2 = [0]*length
        intem_x1 = [0]*length
        final_obj = [0]*length
        intem_loc = [0]*length
        threshold = .95
        locx = [0]*length
        locy = [0]*length
        k = np.ones(len(imk)+1, dtype = np.int)
        lb = frame_env.shape
            
        for index, item in enumerate(imk):
           b[index] = imk[index]
           w, h = b[index].shape[::-1]
           h1 = math.ceil(h/2)
           w1 = math.ceil(w/2)
           
           envir[index] = cv2.copyMakeBorder(frame_env,h1,h1,w1,w1,cv2.BORDER_CONSTANT,0)
           c[index] = cv2.matchTemplate(envir[index], b[index], match_type)
           #cv2.normalize(c[index],c[index],0,1,cv2.NORM_MINMAX,-1)
           #IMportant: max_loc is in the form of (x,y) tuple, i.e. col then row
           min_val, max_val[index], min_loc, max_loc[index] = cv2.minMaxLoc(c[index])
           #maxCorr.insert(index, np.amax(c[index]))
           #maxIndex is a tuple 
           #maxTuple.insert(index, np.unravel_index(c[index].argmax(), c[index].shape))
           final_obj[index] = b[index].shape[::-1]
           intem_loc[index] = max_loc[index]
           intem_x1[index] = x1[index]
           intem_y2[index] = y2[index] 
           b[index] = cv2.resize(obj, (0,0), fx=(k[index]*.002+(index+1)*div/size_range), fy=(k[index]*.002+(index+1)*div/size_range))
           mn[index] = b[index].shape
           #MN is a tuple for the size of the matrix as MxN
           h1 = math.ceil(mn[index][0]/2)
           w1 = math.ceil(mn[index][1]/2)
           y1[index] = max_loc[index][1] + mn[index][0]
           y2[index] = max_loc[index][1] - mn[index][0]
           x1[index] = max_loc[index][0] - mn[index][1]
           x2[index] = max_loc[index][0] + mn[index][1]
           
           if (y2[index] < 0):
               y2[index] = 0
           if (x1[index] < 0):
               x1[index] = 0
           if (x2[index] > lb[1]):
                   x2[index] = lb[1]
           if (y1[index] > lb[0]):
                   y1[index] = lb[0]
           #new_img = img[y2:y1, 0:x2]
           new_frame_env[index] = frame_env[y2[index]:y1[index], x1[index]:x2[index]]
           envir[index] = cv2.copyMakeBorder(new_frame_env[index],h1,h1,w1,w1,cv2.BORDER_CONSTANT,0)
           corrnew[index] = cv2.matchTemplate(envir[index], b[index], match_type)
           min_val_new, max_val_new[index], min_loc_new, max_loc_new[index] = cv2.minMaxLoc(corrnew[index])
           while (max_val_new[index] > (.8*max_val[index]) or k[index] < 10):
               if (k[index] > (div/size_range)/.002):
                  break
               if (max_val_new[index] > max_val[index]):
                  final_obj[index] = b[index].shape[::-1]
                  max_val[index] = max_val_new[index]
                  intem_loc[index] = max_loc_new[index]
                  intem_x1[index] = x1[index]
                  intem_y2[index] = y2[index]
               k[index] = k[index] + 1
               if (max_val_new[index] < threshold):
                  b[index] = cv2.resize(obj, (0,0), fx=(k[index]*.002+(index+1)*div/size_range), fy=(k[index]*.002+(index+1)*div/size_range))
                  mn[index] = b[index].shape
           #MN is a tuple for the size of the matrix as MxN
                  h1 = math.ceil(mn[index][0]/2)
                  w1 = math.ceil(mn[index][1]/2)
                          
                  y1[index] = max_loc[index][1] + mn[index][0]
                          
                  y2[index] = max_loc[index][1] - mn[index][0]
                          
                  x1[index] = max_loc[index][0] - mn[index][1]
                          
                  x2[index] = max_loc[index][0] + mn[index][1]
                          
                  if (y2[index] < 0):
                                
                      y2[index] = 0
                          
                  if (x1[index] < 0):
                                
                        x1[index] = 0
                          
                  if (x2[index] > lb[1]):
                                
                      x2[index] = lb[1]
                         
                  if (y1[index] > lb[0]):
                                
                      y1[index] = lb[0]
           #new_img = img[y2:y1, 0:x2]
                          
                  new_frame_env[index] = frame_env[y2[index]:y1[index], x2[index]:x1[index]]
                  envir[index] = cv2.copyMakeBorder(new_frame_env[index],h1,h1,w1,w1,cv2.BORDER_CONSTANT,0) 
                  corrnew[index] = cv2.matchTemplate(envir[index], b[index], match_type)
                   #cv2.normalize(corrnew[index],corrnew[index],0,1,cv2.NORM_MINMAX,-1)
                  min_val_new, max_val_new[index], min_loc_new, max_loc_new[index] = cv2.minMaxLoc(corrnew[index]) 
                   
                  if (max_val_new[index] > max_val[index]):
                        final_obj[index] = b[index].shape[::-1]
                        intem_loc[index] = max_loc_new[index]
                        intem_x1[index] = x1[index]
                        intem_y2[index] = y2[index]
               if (max_val[index] > threshold):
                 # final_obj[index] = b[index]
                 #  intem_loc[index] = max_loc_new[index]
                 #  intem_x1[index] = x1[index]
                 #  intem_y2[index] = y2[index]
                   break
           locx[index] = intem_loc[index][0] + intem_x1[index]
           locy[index] = intem_loc[index][1] + intem_y2[index]

        return locx, locy, max_val, final_obj

        

    ##
    # @brief finds an object based on a desired color: Developed by Oren Pierce
    # @param frame The frame in which the object needs to be found
    # @param color The color of the desired object
    # @return output returns a color mask (color is white, everything else is black)

    def findObjWColor(self, frame, userColor):
        #define colors and ranges specific color falls under
        red_lower_range = np.array([13, 13, 168], dtype=np.uint8)
        red_upper_range = np.array([101, 101, 242], dtype=np.uint8)

        blue_lower_range = np.array([127, 17, 17], dtype=np.uint8)
        blue_upper_range = np.array([224, 116, 116], dtype=np.uint8)

        green_lower_range = np.array([13, 155, 13], dtype=np.uint8)
        green_upper_range = np.array([111, 232, 111], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if userColor == "red":
            mask = cv2.inRange(frame, red_lower_range, red_upper_range)
        elif userColor == "blue":
            mask = cv2.inRange(frame, blue_lower_range, blue_upper_range)
        elif userColor == "green":
            mask = cv2.inRange(frame, green_lower_range, green_upper_range)
        
        return(mask)


    ##
    # @brief finds average r, g, b value of an image: Developed by Oren Pierce
    # @param frame The frame in which average color is to be found
    # @return output returns average values and image of that color

    def avgColor(self, frame):
        #take average color value of each row of pixels
        average_color_per_row = np.average(frame, axis=0)

        #take the average of that average
        average_color = np.average(average_color_per_row, axis=0)

        #convert the array into uint8 format (for values from 0 to 255)
        average_color = np.uint8(average_color)

        #make a x by y pixel image of the average color
        avg_color_image = np.array([[average_color]*500]*500, np.uint8)

        #write the image file for the average color image
        #cv2.imwrite("avg_color.png", avg_color_image)

        return(avg_color_image)

    ##
    # @brief scans a region of interest (roi) accross an image: Developed by Oren Pierce
    # @param frame The frame the roi is going to scan
    # @param roi_size The side length of a square region of interest
    # @return returns the region of interest

    def roiScan(self, frame, roi_size):

        #convert image to numpy array
        image_data = np.asarray(frame)

        #create a region of interest (roi) with dimentions designated by the user
        roi_height = roi_size
        roi_width = roi_size
        roi = [[]*roi_width]*roi_height
        roi = np.asarray(roi)

        #fill roi with data, repeat across original image 
        for i in range(0, image_data.shape[1] - roi_height, (int)(roi_height / 2)):
            for j in range(0, image_data.shape[1] - roi_width, (int)(roi_width / 2)):
                    print(i)
                    print(j)
                    roi = image_data[i : i + roi_height, j : j + roi_width]
                    cv2.imshow('image', roi)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            else:
                continue
            break    
    
    ##
    # @brief Detects largest contour and draws a box around it
    # @param frame The frame in which a contour will be found
    # @return output Returns a new image with just the largest contour

    def lineIdent(self, frame):
        frame = cv2.medianBlur(frame, 5)

        # Threshold filter to find contours
        ret, thresh = cv2.threshold(frame, 153, 255, 0)
        bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        
        # Find Contours
        q, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # Create Rectangle around largest contour (ROI)
        x, y, w, h = cv2.boundingRect(cnt)
        roi = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        mask = np.zeros(roi.shape,np.uint8)
        mask[y:y+h,x:x+w] = roi[y:y+h,x:x+w]
        output = cv2.bitwise_and(roi, mask)
        
        return output
    
    ##
    # @brief Finds midline of image
    # @param image The final filtered line image
    # @param coordList coordinates of line(2d Numpy array)
    # @param minc The minimum color limit in gray/color
    # @param newcolor The new color to draw the midline in gray/color
    # @return output Returns image with a midline
    def midline(self,image, coordList, minc, newcolor):
        #Sort by y coords
        coordList = coordList[np.argsort(coordList[:,1])]
        
        #Gives every unique y coordinate
        yCoords = np.unique(coordList[:,1])
        
        for coord in yCoords:
            coords = [elem[0] for elem in coordList if elem[1] == coord]
            min = np.amin(coords)
            max = np.amax(coords)
            mid = ((max+min))/2
            image[mid][coord] = newcolor
        output=image
        return output
            
    ##
    # @brief Filters image by shade of gray
    # @param frame The frame to be filtered
    # @param lower The lower color limit in gray
    # @param upper The upper color limit in gray
    # @return output Returns image with only one shade of gray
    def grayfilt(self, frame, lower, upper):
        
        #Sets color filtering threshold
        lower = lower
        upper = upper
        
        #Masks image to find specific color
        mask = cv2.inRange(merge, lower, upper)
        
        #Returns image with only R,G,B visible
        output = cv2.bitwise_and(frame, frame, mask = mask)
        
        return output
    ##
    # @brief Filters image by color
    # @param frame The frame to be filtered
    # @param lower The lower color limit in BGR
    # @param upper The upper color limit in BGR
    # @return output Returns image with only one color
    def colorfilt(self, frame, lower, upper):
        
        #splits into color channels
        b,g,r = cv2.split(frame)
        M = np.maximum(np.maximum(r, g), b)
        r[r < M] = 0
        g[g < M] = 0
        b[b < M] = 0
        
        #Merges max color channels back into the image
        merge = cv2.merge([b, g, r])
        
        #Sets color filtering threshold
        lower = np.array(lower)
        upper = np.array(upper)
        
        #Masks image to find specific color
        mask = cv2.inRange(merge, lower, upper)
        
        #Returns image with only R,G,B visible
        output = cv2.bitwise_and(merge, merge, mask = mask)
        
        return output
        
    ##
    # @brief Orange line detection, developed by Brett Gonzales
    # @param frame The frame to be filtered
    # @return output Filtered frame of largest contour
    def lineDet(self, frame):
        frame = cv2.medianBlur(frame, 5)

        # Threshold filter to find contours
        ret, thresh = cv2.threshold(frame, 153, 255, 0)
        bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        # Find Contours
        q, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # Create Rectangle around largest contour (ROI)
        x, y, w, h = cv2.boundingRect(cnt)
        new = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compare region of interest to original image,
        #  set everything outside the ROI to zero
        mask = np.zeros(new.shape, np.uint8)
        mask[y:y+h, x:x+w] = new[y:y+h, x:x+w]
        out = cv2.bitwise_and(new, mask)

        # Filter for Max RGB values
        b, g, r = cv2.split(out)
        M = np.maximum(np.maximum(r, g), b)
        r[r < M] = 0
        g[g < M] = 0
        b[b < M] = 0

        # Merge back into a maximum RGB image
        image2 = cv2.merge([b, g, r])

        # Set upper and lower limits for colors
        lower = np.array([0, 0, 10])
        upper = np.array([0, 0, 255])

        # Filter out all colors except the tape
        mask = cv2.inRange(image2, lower, upper)

        # Output only tape with black background
        output = cv2.bitwise_and(image2, image2, mask=mask)

        return output
    
    ##
    # @brief Orange line position detection, developed by Brett Gonzales
    # @param detected The detected line to find position
    # @param color The color range for the position detection
    # @return coordList array of pixel coordinates 
    # @return num Total number of red pixels
    def LinePosition(self, detected, color):
        #Convert line to numpy array
        npimg = np.asarray(detected)

        #Find any pixel where it is not black and store a coordinate for that pixel
        coordList = np.argwhere(npimg > color)

        #Find total number of red pixels
        num = len(coordList)

        #Returns array of pixel locations, and total number of red pixels
        return coordList, num
        

    ##
    # @brief Draws the bounding boxes onto a frame
    # @param picks List of bounding boxes
    # @param frame Frame to draw the bounding boxes on
    # @param color RGB tupple representing the color of the bounding boxes
    # @param drawName Whether or not to add the name to the box
    # @param name The name to be displayed
    # @return boxedFrame A copy of frame with the bounding boxes drawn
    def drawBoxes(self,
                  picks,
                  frame,
                  color=(0, 255, 0),
                  drawName=False,
                  name=None):
        boxedFrame = frame.copy()
        for (xA, yA, xB, yB) in picks:
            cv2.rectangle(boxedFrame, (xA, yA), (xB, yB), color, 2)
            if drawName:
                cv2.putText(boxedFrame,
                            name,
                            # self.classifier.getClassNames[boxClass],
                            (xA, yA),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .8,
                            color,
                            2)

        return boxedFrame

    ##
    # @brief Normalizes an image via equalization of histograms
    # @param frame The image to be normalized
    # @return frame The normalized frame
    def normalizeFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(frame)
        channels[0] = cv2.equalizeHist(channels[0])
        frame = cv2.merge(channels)
        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)
        return frame

    ##
    # @brief Finds the corners of a mask
    # @param mask Mask of the image you want to find corners of
    # @pre The mask must be black and white, and only 4 sided polygons will work
    # @return rect Array containing the four corners
    # @return imgCnt The contour of mask used to find the corners
    def getCorners(self, mask):
        (_, cnts, _) = cv2.findContours(mask.copy(),
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        # loop over our contours
        imgCnt = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                imgCnt = approx
                break

        # Find the corners of the contour
        pts = imgCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # print "rect: ", rect
        # cv2.rectangle(thing,
        # (rect[0][0], rect[0][1]),
        # (rect[2][0], rect[2][1]),
        # (100, 100, 100),
        # 2)

        return rect, imgCnt

    ##
    # @brief Uses ray tracing to determine if a point is in a polygon
    # http://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-
    #       2d-point-is-within-a-polygon
    # @param polyPoint List of points of the corners of the polygon
    # @param testPoint The test point that we are checking
    # @return Boolean representing if the point is within the polygon
    def inPolygon(self, polyPoints, testPoint):
        vertx = [polyPoint[0] for polyPoint in polyPoints]
        verty = [polyPoint[1] for polyPoint in polyPoints]
        testx = testPoint[0]
        testy = testPoint[1]

        j = len(vertx) - 1
        inP = False
        for i in range(0, len(vertx)):
            junk = (verty[i] > testy != verty[j] > testy)
            otherjunk = ((vertx[j]-vertx[i]) * (testy-verty[i]) /
                         (verty[j]-verty[i]) + vertx[i])
            if (junk & (testx < otherjunk)):
                inP = not inP
            j = i

        return inP

    ##
    # @brief Draws boxes around found buoys and returns image
    # @param image - The input image with buoys to be drawn over
    # @return image - The final image with boxes drawn over initial image
    def BuoyBoxes(self, image, boxes=0):

        # Get average background color
        avg_color_rows = np.average(image, axis=0)
        avg_color = np.average(avg_color_rows, axis=0)
        avg_color = np.uint8(avg_color)

        # Create image of average color (no use in final product)
        avg_color_img = np.array([[avg_color] * 640] * 360, np.uint8)

        upper_filter = np.array([avg_color[0] + 50, avg_color[1] + 50, avg_color[2] + 50])
        lower_filter = np.array([avg_color[0] - 50, avg_color[1] - 50, avg_color[2] - 50])

        mask = cv2.inRange(image, lower_filter, upper_filter)
        invert = cv2.bitwise_not(mask)
        res = cv2.bitwise_and(image, image, mask=invert)

        frame = cv2.medianBlur(res, 5)

        # Threshold filter to find contours
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find Contours
        q, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the 3 largest contours
        areas = [cv2.contourArea(c) for c in contours]
        i = 0
        cnt = [0, 0, 0]
        max_index = [0, 0, 0]
        while i < 3:
            if i == 2:
                areas[max_index[0]] = 0
                areas[max_index[1]] = 0
            elif i == 1:
                areas[max_index[0]] = 0
            max_index[i] = np.argmax(areas)
            cnt[i] = contours[max_index[i]]

        # Create Rectangle around 3 largest contours (ROI) using average color of middle of contour
            x, y, w, h = cv2.boundingRect(cnt[i])
            k = -5
            redVal = 0
            grnVal = 0
            bluVal = 0
            while k <=5:
                j = -5
                while j <=5:
                    px = image[math.floor(y+0.5*h)+k, math.floor(x+0.5*w)+j]
                    redVal += int(px[0])
                    grnVal += int(px[1])
                    bluVal += int(px[2])
                    j += 1
                k += 1
            redVal /= 121
            grnVal /= 121
            bluVal /= 121
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (redVal, grnVal, bluVal), 2)
            midx = x + 0.5*w
            midy = y + 0.5*h
            buoyString = 'buoy #'
            #buoyString = 'buoy #' + str(boxes.check((midx, midy), (redVal, grnVal, bluVal)))
            image = cv2.putText(image, buoyString, (x, y+h),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 0, cv2.LINE_AA)
            i += 1
        return image

class buoyTracker:
    location = []
    color = []
    ##
    # @brief Does nothing important for the time being.  Will add some
    #        functionality later
    def __init__(self):
        # Dummy variable, not used for anything
        self.useme = True

    ##
    # @brief base call function to track buoys
    # @param location - midpoint of buoy to track
    # @param color - RGB color of buoy to track
    # @return match - returns the buoy number that has been tracked
    def check(self, location, color):
        closestLoc = self.checkLoc(location)
        colorMatch = self.checkColor(color)
        if len(self.location) < 3 or closestLoc == -1 or colorMatch == -1 or closestLoc != colorMatch:
            self.addBuoy(location, color)
            match = len(self.location) - 1
        else:
            match = closestLoc
        self.update(location, color, match)
        return match

    ##
    # @brief Finds buoy with closest location to last known buoys
    # @param location - midpoint of buoy to track
    # @return locMatch - index of buoy that has been tracked
    def checkLoc(self, location):
        locMatch = -1
        closest = -1
        for c in self.location:
            x1, y1 = c
            x2, y2 = location
            distance = math.hypot(x1 - x2, y1 - y2)
            if closest == -1:
                closest = distance
                locMatch = self.location.index(c)
            elif closest > distance:
                closest = distance
                locMatch = self.location.index(c)
        return locMatch

    ##
    # @brief Finds buoy with closest color to last known buoys
    # @param color - RGB color of buoy to track
    # @return colorMatch - index of buoy that has been tracked
    def checkColor(self, color):
        colorMatch = -1
        closest = 250
        for c in self.color:
            difference = abs(color[0] - c[0]) + abs(color[1] - c[1]) + abs(color[2] - c[2])
            if difference < closest:
                closest = difference
                colorMatch = self.color.index(c)
        return colorMatch

    ##
    # @brief Adds a new buoy to class function to be searched
    # @param location - midpoint of buoy to track
    # @param color - RGB color of buoy to track
    def addBuoy(self, location, color):
        self.location.append(location)
        self.color.append(color)

    ##
    # @brief updates buoy's new location and color
    # @param location - midpoint of buoy to track
    # @param color - RGB color of buoy to track
    # @param index - index of buoy to be updated
    def update(self, location, color, index):
        self.location[index] = location
        self.color[index] = color
