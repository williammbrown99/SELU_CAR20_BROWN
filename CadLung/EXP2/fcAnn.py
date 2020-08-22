import NasFcAnn

testModel = NasFcAnn.NasFcAnn(dataPath='data_balanced_6Slices_1orMore_xy.bin', name='xySlice', normalize='zs', regRate=0.001)
#!!!must include data path!!!
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

print('done!')