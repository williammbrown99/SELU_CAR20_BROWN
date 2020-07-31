import NasFcAnn

testModel = NasFcAnn.NasFcAnn(name='zscore', normalize='zs', regRate=0.001)
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