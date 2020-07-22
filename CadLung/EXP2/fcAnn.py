import NasFcAnn

testModel = NasFcAnn.NasFcAnn(name='z-score', normalize='zs')   #Options: {Default: 'none', 'ra', 'zs', 'pr'}

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