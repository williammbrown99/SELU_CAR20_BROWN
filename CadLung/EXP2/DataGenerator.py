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

#Paths
DataSet_xy = 'data_balanced_6Slices_1orMore_xy.bin'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz.bin'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz.bin'

xySliceTrainPredPath = 'OUTPUT/xySliceTrainPredictions.tf'
xySliceTestPredPath = 'OUTPUT/xySliceTestPredictions.tf'
xzSliceTrainPredPath = 'OUTPUT/xzSliceTrainPredictions.tf'
xzSliceTestPredPath = 'OUTPUT/xzSliceTestPredictions.tf'
yzSliceTrainPredPath = 'OUTPUT/yzSliceTrainPredictions.tf'
yzSliceTestPredPath = 'OUTPUT/yzSliceTestPredictions.tf'

xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'
xzPlaneInputPath = 'CadLung/INPUT/PLANE/xzPlaneInput.npz'
yzPlaneInputPath = 'CadLung/INPUT/PLANE/yzPlaneInput.npz'

xyPlaneTrainPredPath = 'OUTPUT/xyPlaneTrainPredictions.tf'
xyPlaneTestPredPath = 'OUTPUT/xyPlaneTestPredictions.tf'
xzPlaneTrainPredPath = 'OUTPUT/xzPlaneTrainPredictions.tf'
xzPlaneTestPredPath = 'OUTPUT/xzPlaneTestPredictions.tf'
yzPlaneTrainPredPath = 'OUTPUT/yzPlaneTrainPredictions.tf'
yzPlaneTestPredPath = 'OUTPUT/yzPlaneTestPredictions.tf'

volumeInputPath = 'CadLung/INPUT/VOLUME/volumeInput.npz'
###

# Convert labels to 3d
PredictionDataProcess = PredictionDataProcess()
trainLabel = PredictionDataProcess.convertTrainingLabels(DataSet_xy)[0]   #Training Labels (420)
testLabel = PredictionDataProcess.convertTrainingLabels(DataSet_xy)[1]    #Test Labels    (182)
###

# XY Plane Pre Process
xyPlaneTrainSet = PredictionDataProcess.sliceToPlane(xySliceTrainPredPath)      #Training Set
xyPlaneTestSet = PredictionDataProcess.sliceToPlane(xySliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(xyPlaneInputPath, xyPlaneTrainSet, trainLabel, xyPlaneTestSet, testLabel)
###

# XZ Plane Pre Process
xzPlaneTrainSet = PredictionDataProcess.sliceToPlane(xzSliceTrainPredPath)      #Training Set
xzPlaneTestSet = PredictionDataProcess.sliceToPlane(xzSliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(xzPlaneInputPath, xzPlaneTrainSet, trainLabel, xzPlaneTestSet, testLabel)
###

# YZ Plane Pre Process
yzPlaneTrainSet = PredictionDataProcess.sliceToPlane(yzSliceTrainPredPath)      #Training Set
yzPlaneTestSet = PredictionDataProcess.sliceToPlane(yzSliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(yzPlaneInputPath, yzPlaneTrainSet, trainLabel, yzPlaneTestSet, testLabel)
###

# Volume Pre Process
volumeTrainSet = PredictionDataProcess.planeToVolume(xyPlaneTrainPredPath,
                                                     xzPlaneTrainPredPath,
                                                     yzPlaneTrainPredPath)     #Training Set

volumeTestSet = PredictionDataProcess.planeToVolume(xyPlaneTestPredPath,
                                                    xzPlaneTestPredPath,
                                                    yzPlaneTestPredPath)       #Test Set

PredictionDataProcess.exportInputs(volumeInputPath, volumeTrainSet, trainLabel, volumeTestSet, testLabel)
###

print('done!')