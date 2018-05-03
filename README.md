# New multicamera_framework for RoboSub 2017/18o
all test files are located on the google drive. Path: RoboSub2-->Electrical-->Code for Sensors-->Cameras
put test_files into directory: vision-->tests
if any needed test files are not on the drive, see line 12 of this file
this path also leads to how to install opencv w/ contrib packages and anaconda
in tests, test_BuoyVideo.py does not work. Need to debug
in tests, test_videoLineDet.py does not work. Need to debug
read the FIXME comments in all tests and in vision_tools
read the TODO comments in all tests and vision_tools: 
classifier child class svmClassifier is still in progress (under vision-->vision-->classifiers)

NOTE: an SVM (support vector machine) is a type of classifier that can distinguish between different things.
a binary SVM distinguishes between 2 things (i.e. a certain object and a background). a basic understanding of
guided learning will be needed to use an SVM. Guided learning means that a computer is "trained" to know what something
is based on something it has "seen" many many times.
Topics to study: feature vector, opencv classifier, receiver operating characteristic curves (ROC)


It there are any questions, please contact Oren Pierce at oren.pierce@yahoo.com
