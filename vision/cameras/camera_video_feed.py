# camera_video_feed.py - This is a 'test' camera that will take in a video feed
#                      and act as a camera
from .camera import Camera
import cv2


class videoFeedCamera(Camera):
    ##
    # @param fileLocation The absolute file path to the video
    # @post An openCV connection to the video file
    def __init__(self, connType=99, name="Image Feed", debug=None):
        # use the IPaddress as the video file location
        Camera.__init__(self, connType, name, debug)
        self.cap = cv2.VideoCapture(debug)
        print("Initialized camera.", self.__str__())

    ##
    # @return Processed frame that was captured from the camera
    def getFrame(self):
        success, frame = self.cap.read()
        if (success):
            cv2.imshow('ting',frame)
            print("shape: ", frame.shape)
            cv2.waitKey(0)

            return frame
        else:
            print('failure')
