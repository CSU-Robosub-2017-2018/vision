'''
classifier.py - Extend this class through inheritance when adding different
classifiers. For compliance with the framework, the extended classifier class MUST
implement at least the extractFeatures function and call the parent __init__
function during its __init__ call

'''


class Classifier:
    ##
    # @param classifierType The type of classifier being used.  0=SVM
    # @param name Name of the classifier
    # @post An openCV classifierection to the camera
    def __init__(self, classifierType=0, name="default"):
        self.classifierType = classifierType
        # Decode of the classifierection type
        if self.classifierType == 0:
            self.classifierTypeStr = "SVM"
                #for additional classifier types used in the future, add them here (example below)
        #elif self.classifierType == 1:
        #    self.classifierTypeStr = "(insert classifer type)"
        else:
            self.classifierTypeStr = 'DEMONS!!!'

        self.name = name

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getclassifierType(self):
        return self.classifierType

    def setclassifierType(self, classifierType):
        self.classifierType = classifierType

    def getclassifierTypeStr(self):
        return self.classifierTypeStr

    def setclassifierTypeStr(self, classifierTypeStr):
        self.classifierTypeStr = classifierTypeStr

    def __str__(self):
        return "Classifier name: %s, Classifier type: %s" % (
        self.name, self.classifierTypeStr)

