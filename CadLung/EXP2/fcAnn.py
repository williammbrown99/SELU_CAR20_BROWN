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
- File to train and test models
'''
import NasFcAnn

DataSet_xy = 'data_balanced_6Slices_1orMore_xy.bin'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz.bin'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz.bin'

xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'
xzPlaneInputPath = 'CadLung/INPUT/PLANE/xzPlaneInput.npz'
yzPlaneInputPath = 'CadLung/INPUT/PLANE/yzPlaneInput.npz'

volumeInputPath = 'CadLung/INPUT/VOLUME/volumeInput.npz'

testModel = NasFcAnn.NasFcAnn(dataPath=volumeInputPath, type='volume',
                              name='volume', regRate=0.1, positiveRegion='n',
                              normalize='zs')
#!!!must include data path!!!
#type Options: {Default: 'slice', 'plane', 'volume'}
#positiveRegion options: {Default: 'n', 'y'}
#name options: {Default: 'none'}
#normalize options: {Default: 'none', 'ra', 'zs'}
#regRate options: {Default: 0.001}

testModel.exportParam()
testModel.loadData()
testModel.exportData()
testModel.doPreProcess()
testModel.exportPreProcData()
testModel.setUpModelTrain()
testModel.exportModel()
testModel.loadModel()
testModel.testModel()
testModel.exportPredict()
testModel.evaluate()
testModel.exportTestPerf()
testModel.visualPerf()
testModel.exportChart()

#function to export training predictions to next model
testModel.exportTrainPred()

#function to export model weights for reporting
testModel.exportModelWeights()

#function to export training error for reporting
testModel.exportTrainingError()

print('done!')