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
import NasFcAnn
import keras
import numpy as np

#Parameters
paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT/')
xySliceTrainPredPath = 'OUTPUT/xySliceTrainPredictions.tf'
xySliceTestPredPath = 'OUTPUT/xySliceTestPredictions.tf'

voiDataPath = 'data_balanced_6Slices_1orMore_xy.bin'
xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'
###

#Functions
def readTrainData(path):
    '''Read XY Slice Model Predictions'''
    numSlices = 6
    with open(paths[1]+path, mode='r') as file:
        xySlicePred = file.readlines()

    for i in range(0, len(xySlicePred)):
        xySlicePred[i] = float(xySlicePred[i])

    #Reshaping feature vector
    pred = [xySlicePred[x:x+numSlices] for x in range(0, len(xySlicePred), numSlices)]

    return pred
#

def convertTrainingLabels(path):
    '''Convert Slice train labels for Plane Model'''
    numSlices = 6
    dataset_filepath = paths[0]+path
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

def exportPlaneInputs(path, trainSet, trainLabel, testSet, testLabel):
    '''Export Imports for Plane Model'''
    np.savez(path, trainSet, trainLabel, testSet, testLabel)
    file = np.load(path)
    print(file['arr_0'])
###

#Main Code
xyPlaneTrainSet = readTrainData(xySliceTrainPredPath)       #Training Set
xyPlaneTrainLabel = convertTrainingLabels(voiDataPath)[0]   #Training Labels
xyPlaneTestSet = readTrainData(xySliceTestPredPath)         #Test Set
xyPlaneTestLabel = convertTrainingLabels(voiDataPath)[1]    #Test Labels

exportPlaneInputs(xyPlaneInputPath, xyPlaneTrainSet, xyPlaneTrainLabel, xyPlaneTestSet, xyPlaneTestLabel)

print('done!')
###