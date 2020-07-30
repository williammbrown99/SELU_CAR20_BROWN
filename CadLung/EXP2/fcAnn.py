import NasFcAnn

testModel = NasFcAnn.NasFcAnn(name='Range', normalize='ra')   #Options: {Default: 'none', 'ra', 'zs'}

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