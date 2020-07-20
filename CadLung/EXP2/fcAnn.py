import NasFcAnn

testModel = NasFcAnn.NasFcAnn()

testModel.loadData()
testModel.doPreProcess()
testModel.setUpModelTrain()
testModel.exportModel()
testModel.testModel()
testModel.evaluate()
testModel.visualPerf()

print('done!')