import NasFcAnn

xySliceInputPath = 'data_balanced_6Slices_1orMore_xy.bin'
xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'

testModel = NasFcAnn.NasFcAnn(dataPath=xyPlaneInputPath, type='plane',
                              name='xyPlane', regRate=0.001, positiveRegion='n',
                              normalize='none')
#!!!must include data path!!!
#type Options: {Default: 'slice', 'plane'}
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
testModel.testModel()
testModel.exportPredict()
testModel.evaluate()
testModel.exportTestPerf()
testModel.visualPerf()
testModel.exportChart()

#function to export training predictions to next model
testModel.exportTrainPred()

print('done!')