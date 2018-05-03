# svmClassifier.py - This is a 'test' classifier that will take in a single feature vector
#                      and classify an object in a frame
import os
import cv2
# import imutils
from .classifier import Classifier


class svmClassifier(Classifier):
    ##
    # @brief initialize svm classifier
    # @param classifierType the type of classifier specified in the Classifier Class
    # @param name name of the instantiated classifier
    def __init__(self, classifierType=0, name="testSVM", vision_tools = None):
        if vision_tools is not None:
            self.vision_tools = vision_tools
        else:
            self.vision_tools = None
        Classifier.__init__(self, classifierType, name)
        print("Initialized classifier.", self.__str__())

    ##
    # @brief set up parameter for initial, untrained svm
    # @param C one of the parameters needed by opencv for svms
    # @param gamma one of the parameters needed by opencv for svms
    # @return model untrained base model svm classifier
    def svmInit(self, C=12.5, gamma=0.50625, trained_model = None):
        #initialize SVM parameters
        print('Initializing SVM model with given parameters ... ')
        if trained_model is not None:
            #load trained_model
            model = 1
        else:
            model = cv2.ml.SVM_create()
            model.setGamma(gamma)
            model.setC(C)
            model.setKernel(cv2.ml.SVM_RBF)
            model.setType(cv2.ml.SVM_C_SVC)
        return model

    ##
    # @brief load data given what object is needed to be classified
    # @param userMask the desired object shape to be classified
    # @param userColor the desire object color to be classified
    # @param SZ
    # @param BIN_N number of bins
    # @return trained_model svm model for feature vector to be compared to
    def loadData(self, userMask, userColor, SZX=64, SZY=128 BIN_N=10):
        #determine how the svm will be trained
        #training images are in files associated with what is needed to be classified
        #for example, an orange circle has its own set of training images. Additional training files can be made as needed
        if userMask == 0 & userColor == 0:
            train_features_file = "/home/oren/vision/tests/test_files/orange_ball_training"
        #elif userMask == 1 & userColor == 1:
            #train_features_file = /path/green_square_training(example)
        else:
            print("INPUT PARAMETERS NOT DEFINED")

        print('Defining HoG Parameters ... ')
        winSize = (20,20)
        blockSize = (8,8)
        blockStride = (4,4)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True

        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

        print('Loading SVM Training Data ... ')
        #cycle through training images in folder and split image data into training and testing data
        for file in os.listdir(train_features_file):
            #FIXME replace directory with train_features file
            filename = os.fsdecode(file)
            dirname = os.fsdecode(directory)
            filefile = os.path.join(dirname, filename)
            subimage = cv2.imread(str(filefile))
            subimage = cv2.resize(subimage, (SZX, SZY))

            curr_feat, feat_size = tools.getAllHog(subimage,
                                                stridex,
                                                stridey,
                                                szx,
                                                szy,
                                                cell_size,
                                                block_size,
                                                nbins)
            curr_feat = curr_feat.reshape((feat_size, 1))

            trainfeats.append(curr_feat)
            trainlabels.append(1)
            trainimages.append(subimage)


    ##
    # @brief train SVM model
    # @param model untrained svm model
    # @param img
    # @param objects_deskewed
    # @return trained_model trained SVM model
    def trainSVM(model, img, objects_deskewed):



        print('Calculating HoG descriptor for each image ... ')
        hog_descriptors = []
        #FIXME this is used inf vision_tools is not None
        RBG = self.vision_tools.getAverageColor(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for img in objects_deskewed:
            hog_descriptors.append(hog.compute(img))
        #FIXME remove line below and put into somethingx1 matrix
        hog_descriptors = np.squeeze(hog_descriptors)

        print('Splitting data into training (90%) and test set (10%) ... ')
        train_n = int(0.9*len(hog_descriptors))
        objects_train, objects_test = np.split(objects_deskewed, [train_n])
        hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])

        print('Training SVM model ... ')
        trained_model = model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

        return trained_model, objects_test, hog_descriptors_test, labels_test)

    ##
    # @brief evaluate trained SVM model to classify object
    # @param trained_model
    # @param objects_test
    # @param hog_descriptors_test
    # @param labels_test
    def classify(trained_model, objects_test, hog_descriptors_test, labels_test):
        print('Evaluating Model ... ')
        predictions = trained_model.predict(trained_model, hog_descriptors_test)
        accuracy = (labels == predictions).mean()
        print('Percentage Accuracy: %.2f %%' % (accuracy*100))
        confusion = np.zeros((10, 10), np.int32)
        for i, j in zip(labels_test, predictions):
            confusion[int(i), int(j)] += 1
        print('confusion matrix:')
        print(confusion)

        final = []
        for img, flag in zip(objects_test, predictions == labels_test):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BRG)
            if not flag:
                img[...,:2] = 0
            final.append(img)

        return final




f










