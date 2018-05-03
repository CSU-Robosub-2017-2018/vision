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

+# RoboSub Vision Tools 2017/18
+Someone needs to fill this out with instructions on how to install dependencies, run this script, add test cases, run test cases, add new components, why this repo matters, etc.  The multicamera_framework REDME.md can be used as a clear reference on how to fill this file out. General HTML codes work for normal things (<B>bold</B>, <I>italic</I>, etc).  Check https://www.stack.nl/~dimitri/doxygen/manual/markdown.html for more style codes and such.
+
+DO THE DAMN DOCUMENTATION!!!
+
+<B>NOTE</B>:  You will need to install the doxygen and graphviz packages in order to run doxygen and generate the outputs.  You can do so with the following command on linux (windows users can blow me):
+
+-  sudo apt-get install doxygen graphviz
+
+Then cd into the doc directory and run the following:
+  
+-  doxygen config.dox
+
+After the command runs, there will be an html/ directory.  Open html/index.html in your favorite web browser.
+
+In general use case, the documentation does <I>not</I> get checked into the repository.  The idea being that code is changing so quickly, that if someone wants up to date documentation then they can download the repository themselves and run doxygen to generate the docs.
+
+# Thanks. Your robot overlord, Bender.
