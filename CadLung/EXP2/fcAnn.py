import NasFcAnn

testModel = NasFcAnn.NasFcAnn(normalize='ra')   #Options: {Default: 'none', 'ra', 'zs', 'pr'}

testModel.loadData()
testModel.doPreProcess()
testModel.setUpModelTrain()
testModel.exportModel()
testModel.testModel()
testModel.evaluate()
testModel.visualPerf()
testModel.exportChart()

print('done!')