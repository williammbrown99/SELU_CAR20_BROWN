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
- Main File
'''
import NasFcAnn
import PredictionDataProcess
import keras
import numpy as np

from keras.metrics import AUC

#Parameters
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
PredictionDataProcess = PredictionDataProcess.PredictionDataProcess()
trainLabel = PredictionDataProcess.convertTrainingLabels(DataSet_xy)[0]   #Training Labels (420)
testLabel = PredictionDataProcess.convertTrainingLabels(DataSet_xy)[1]    #Test Labels    (182)
###

# XY Slice Model
xySliceModel = NasFcAnn.NasFcAnn(dataPath=DataSet_xy, type='slice',
                              name='xySlice', regRate=0.001, positiveRegion='y',
                              normalize='zs')

xySliceModel.loadData()
xySliceModel.doPreProcess()

xySliceModel.loadModel()
print('XY Slice Model Evaluation:')
xySliceModel.evaluate()
###

# XY Plane Pre Process
xyPlaneTrainSet = PredictionDataProcess.sliceToPlane(xySliceTrainPredPath)      #Training Set
xyPlaneTestSet = PredictionDataProcess.sliceToPlane(xySliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(xyPlaneInputPath, xyPlaneTrainSet, trainLabel, xyPlaneTestSet, testLabel)
###

# XY Plane Model
xyPlaneModel = NasFcAnn.NasFcAnn(dataPath=xyPlaneInputPath, type='plane',
                              name='xyPlane', regRate=0.001, positiveRegion='n',
                              normalize='zs')

xyPlaneModel.loadData()
xyPlaneModel.doPreProcess()

xyPlaneModel.loadModel()
print('\nXY Plane Model Evaluation:')
xyPlaneModel.evaluate()
###

# XZ Slice Model
xzSliceModel = NasFcAnn.NasFcAnn(dataPath=DataSet_xz, type='slice',
                              name='xzSlice', regRate=0.001, positiveRegion='y',
                              normalize='zs')

xzSliceModel.loadData()
xzSliceModel.doPreProcess()

xzSliceModel.loadModel()
print('\nXZ Slice Model Evaluation:')
xzSliceModel.evaluate()
###

# XZ Plane Pre Process
xzPlaneTrainSet = PredictionDataProcess.sliceToPlane(xzSliceTrainPredPath)      #Training Set
xzPlaneTestSet = PredictionDataProcess.sliceToPlane(xzSliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(xzPlaneInputPath, xzPlaneTrainSet, trainLabel, xzPlaneTestSet, testLabel)
###

# XZ Plane Model
xzPlaneModel = NasFcAnn.NasFcAnn(dataPath=xzPlaneInputPath, type='plane',
                              name='xzPlane', regRate=0.001, positiveRegion='n',
                              normalize='zs')

xzPlaneModel.loadData()
xzPlaneModel.doPreProcess()

xzPlaneModel.loadModel()
print('\nXZ Plane Model Evaluation:')
xzPlaneModel.evaluate()
###

# YZ Slice Model
yzSliceModel = NasFcAnn.NasFcAnn(dataPath=DataSet_yz, type='slice',
                              name='yzSlice', regRate=0.001, positiveRegion='y',
                              normalize='zs')

yzSliceModel.loadData()
yzSliceModel.doPreProcess()

yzSliceModel.loadModel()
print('\nYZ Slice Model Evaluation:')
yzSliceModel.evaluate()
###

# YZ Plane Pre Process
yzPlaneTrainSet = PredictionDataProcess.sliceToPlane(yzSliceTrainPredPath)      #Training Set
yzPlaneTestSet = PredictionDataProcess.sliceToPlane(yzSliceTestPredPath)        #Test Set

PredictionDataProcess.exportInputs(yzPlaneInputPath, yzPlaneTrainSet, trainLabel, yzPlaneTestSet, testLabel)
###

# YZ Plane Model
yzPlaneModel = NasFcAnn.NasFcAnn(dataPath=yzPlaneInputPath, type='plane',
                              name='yzPlane', regRate=0.001, positiveRegion='n',
                              normalize='zs')

yzPlaneModel.loadData()
yzPlaneModel.doPreProcess()

yzPlaneModel.loadModel()
print('\nYZ Plane Model Evaluation:')
yzPlaneModel.evaluate()
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

# Volume Model
volumeModel = NasFcAnn.NasFcAnn(dataPath=volumeInputPath, type='volume',
                              name='volume', regRate=0.001, positiveRegion='n',
                              normalize='zs')

volumeModel.loadData()
volumeModel.doPreProcess()

volumeModel.loadModel()
print('\nVolume Model Evaluation:')
volumeModel.evaluate()
###

print('done!')
