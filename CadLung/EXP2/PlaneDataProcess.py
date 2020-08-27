'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: August 21 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
- Read the coding standarts and best practice notes.
~~~~ HELP  ~~~~
- Enter your notes to help to understand your code here
'''
import numpy as np

class PlaneDataProcess(object):
    #Attributes
    paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT/')
    ###

    def __init__(self, **kwarg):
        pass
    #

    def readTrainData(self, path):
        '''Read XY Slice Model Predictions'''
        numSlices = 6
        with open(self.paths[1]+path, mode='r') as file:
            xySlicePred = file.readlines()

        for i in range(0, len(xySlicePred)):
            xySlicePred[i] = float(xySlicePred[i])

        #Reshaping feature vector
        pred = [xySlicePred[x:x+numSlices] for x in range(0, len(xySlicePred), numSlices)]

        return pred
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

    def exportPlaneInputs(self, path, trainSet, trainLabel, testSet, testLabel):
        '''Export Imports for Plane Model'''
        np.savez(path, trainSet, trainLabel, testSet, testLabel)
        file = np.load(path)
    ###
###