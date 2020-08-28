'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: August 27 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
- Read the coding standarts and best practice notes.
~~~~ HELP  ~~~~
- Class to preprocess data for plane models
'''
import numpy as np

class PredictionDataProcess(object):
    #Attributes
    paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT/')
    ###

    def __init__(self, **kwarg):
        pass
    #

    def sliceToPlane(self, path):
        '''Read XY Slice Model Predictions'''
        numSlices = 6
        with open(self.paths[1]+path, mode='r') as file:
            slicePred = file.readlines()

        for i in range(0, len(slicePred)):
            slicePred[i] = float(slicePred[i])

        #Reshaping feature vector
        planeInputs = [slicePred[x:x+numSlices] for x in range(0, len(slicePred), numSlices)]

        return planeInputs
    #

    def convertTrainingLabels(self, path):
        '''Convert Slice train labels for Plane Model'''
        numSlices = 6
        dataset_filepath = self.paths[0]+path
        with open(dataset_filepath, 'rb') as f2:
              train_set_all = np.load(f2)
              train_label_all = np.load(f2)
              test_set_all = np.load(f2)
              test_label_all = np.load(f2)
    
        #Reshaping Training Labels
        trainLabels = [train_label_all[x] for x in range(0, len(train_label_all), numSlices)]
        testLabels = [test_label_all[x] for x in range(0, len(test_label_all), numSlices)]

        return [trainLabels, testLabels]
    #

    def planeToVolume(self, path1, path2, path3):
        '''Read XY Slice Model Predictions'''
        # Reading Files
        with open(self.paths[1]+path1, mode='r') as file:
            xyPlanePred = file.readlines()
        with open(self.paths[1]+path2, mode='r') as file:
            xzPlanePred = file.readlines()
        with open(self.paths[1]+path3, mode='r') as file:
            yzPlanePred = file.readlines()
        
        # Converting to floats
        for i in range(0, len(xyPlanePred)):
            xyPlanePred[i] = float(xyPlanePred[i])
        for j in range(0, len(xzPlanePred)):
            xzPlanePred[j] = float(xzPlanePred[j])
        for k in range(0, len(yzPlanePred)):
            yzPlanePred[k] = float(yzPlanePred[k])

        #Reshaping feature vector
        volumeInputs = [[xyPlanePred[x], xzPlanePred[x], yzPlanePred[x]] for x in range(0, len(xyPlanePred))]

        return volumeInputs
    #

    def exportInputs(self, path, trainSet, trainLabel, testSet, testLabel):
        '''Export Imports for Plane Model'''
        np.savez(path, trainSet, trainLabel, testSet, testLabel)
        file = np.load(path)
    ###
###