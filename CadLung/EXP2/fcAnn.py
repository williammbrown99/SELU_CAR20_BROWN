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
!!!must include data path!!!
type Options: {Default: 'slice', 'plane', 'volume'}
positiveRegion options: {Default: 'n', 'y'}
name options: {Default: 'none'}
normalize options: {Default: 'none', 'ra', 'zs'}
regRate options: {Default: 0.001}
'''
import NasFcAnn
import DataGenerator
from datetime import datetime
#timing code
startTime = datetime.now()

DataSet_xy = 'data_balanced_6Slices_1orMore_xy.bin'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz.bin'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz.bin'

xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'
xzPlaneInputPath = 'CadLung/INPUT/PLANE/xzPlaneInput.npz'
yzPlaneInputPath = 'CadLung/INPUT/PLANE/yzPlaneInput.npz'

volumeInputPath = 'CadLung/INPUT/VOLUME/volumeInput.npz'

#DATA GENERATOR
DataGen = DataGenerator.DataGenerator()

#XY SLICE MODEL
print('\nXY SLICE MODEL')
xySlice = NasFcAnn.NasFcAnn(dataPath=DataSet_xy, type='slice',
                              name='xySlice', regRate=0.1, positiveRegion='y',
                              normalize='zs')

xySlice.exportParam()
xySlice.loadData()
xySlice.exportData()
xySlice.doPreProcess()
xySlice.exportPreProcData()
xySlice.setUpModelTrain()
xySlice.exportModel()
xySlice.loadModel()
xySlice.testModel()
xySlice.exportPredict()
xySlice.evaluate()
xySlice.exportTestPerf()
#xySlice.visualPerf()
#xySlice.exportChart()

#function to export training predictions to next model
xySlice.exportTrainPred()

#function to export model weights for reporting
xySlice.exportModelWeights()

#function to export training error for reporting
xySlice.exportTrainingError()
#########

#XZ SLICE MODEL
print('\nXZ SLICE MODEL')
xzSlice = NasFcAnn.NasFcAnn(dataPath=DataSet_xz, type='slice',
                              name='xzSlice', regRate=0.1, positiveRegion='y',
                              normalize='zs')

xzSlice.exportParam()
xzSlice.loadData()
xzSlice.exportData()
xzSlice.doPreProcess()
xzSlice.exportPreProcData()
xzSlice.setUpModelTrain()
xzSlice.exportModel()
xzSlice.loadModel()
xzSlice.testModel()
xzSlice.exportPredict()
xzSlice.evaluate()
xzSlice.exportTestPerf()
#xzSlice.visualPerf()
#xzSlice.exportChart()

#function to export training predictions to next model
xzSlice.exportTrainPred()

#function to export model weights for reporting
xzSlice.exportModelWeights()

#function to export training error for reporting
xzSlice.exportTrainingError()
#####

#YZ SLICE MODEL
print('\nYZ SLICE MODEL')
yzSlice = NasFcAnn.NasFcAnn(dataPath=DataSet_yz, type='slice',
                              name='yzSlice', regRate=0.1, positiveRegion='y',
                              normalize='zs')

yzSlice.exportParam()
yzSlice.loadData()
yzSlice.exportData()
yzSlice.doPreProcess()
yzSlice.exportPreProcData()
yzSlice.setUpModelTrain()
yzSlice.exportModel()
yzSlice.loadModel()
yzSlice.testModel()
yzSlice.exportPredict()
yzSlice.evaluate()
yzSlice.exportTestPerf()
#yzSlice.visualPerf()
#yzSlice.exportChart()

#function to export training predictions to next model
yzSlice.exportTrainPred()

#function to export model weights for reporting
yzSlice.exportModelWeights()

#function to export training error for reporting
yzSlice.exportTrainingError()
#####

#CREATING PLANE INPUT DATA
print()
DataGen.createPlaneData()

#XY PLANE MODEL
print('\nXY PLANE MODEL')
xyPlane = NasFcAnn.NasFcAnn(dataPath=xyPlaneInputPath, type='plane',
                              name='xyPlane', regRate=0.1, positiveRegion='n',
                              normalize='zs')

xyPlane.exportParam()
xyPlane.loadData()
xyPlane.exportData()
xyPlane.doPreProcess()
xyPlane.exportPreProcData()
xyPlane.setUpModelTrain()
xyPlane.exportModel()
xyPlane.loadModel()
xyPlane.testModel()
xyPlane.exportPredict()
xyPlane.evaluate()
xyPlane.exportTestPerf()
#xyPlane.visualPerf()
#xyPlane.exportChart()

#function to export training predictions to next model
xyPlane.exportTrainPred()

#function to export model weights for reporting
xyPlane.exportModelWeights()

#function to export training error for reporting
xyPlane.exportTrainingError()
#########

#XZ PLANE MODEL
print('\nXZ PLANE MODEL')
xzPlane = NasFcAnn.NasFcAnn(dataPath=xzPlaneInputPath, type='plane',
                              name='xzPlane', regRate=0.1, positiveRegion='n',
                              normalize='zs')

xzPlane.exportParam()
xzPlane.loadData()
xzPlane.exportData()
xzPlane.doPreProcess()
xzPlane.exportPreProcData()
xzPlane.setUpModelTrain()
xzPlane.exportModel()
xzPlane.loadModel()
xzPlane.testModel()
xzPlane.exportPredict()
xzPlane.evaluate()
xzPlane.exportTestPerf()
#xzPlane.visualPerf()
#xzPlane.exportChart()

#function to export training predictions to next model
xzPlane.exportTrainPred()

#function to export model weights for reporting
xzPlane.exportModelWeights()

#function to export training error for reporting
xzPlane.exportTrainingError()
#########

#YZ PLANE MODEL
print('\nYZ PLANE MODEL')
yzPlane = NasFcAnn.NasFcAnn(dataPath=yzPlaneInputPath, type='plane',
                              name='yzPlane', regRate=0.1, positiveRegion='n',
                              normalize='zs')

yzPlane.exportParam()
yzPlane.loadData()
yzPlane.exportData()
yzPlane.doPreProcess()
yzPlane.exportPreProcData()
yzPlane.setUpModelTrain()
yzPlane.exportModel()
yzPlane.loadModel()
yzPlane.testModel()
yzPlane.exportPredict()
yzPlane.evaluate()
yzPlane.exportTestPerf()
#yzPlane.visualPerf()
#yzPlane.exportChart()

#function to export training predictions to next model
yzPlane.exportTrainPred()

#function to export model weights for reporting
yzPlane.exportModelWeights()

#function to export training error for reporting
yzPlane.exportTrainingError()
#########

#CREATING VOLUME INPUT DATA
print()
DataGen.createVolumeData()

#VOLUME MODEL
print('\nVOLUME MODEL')
volume = NasFcAnn.NasFcAnn(dataPath=volumeInputPath, type='volume',
                              name='volume', regRate=0.1, positiveRegion='n',
                              normalize='zs')

volume.exportParam()
volume.loadData()
volume.exportData()
volume.doPreProcess()
volume.exportPreProcData()
volume.setUpModelTrain()
volume.exportModel()
volume.loadModel()
volume.testModel()
volume.exportPredict()
volume.evaluate()
volume.exportTestPerf()
volume.visualPerf()
volume.exportChart()

#Printing Execution Time
print('\nExecution time: {}'.format(datetime.now() - startTime))