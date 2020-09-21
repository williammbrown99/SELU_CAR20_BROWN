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

#Parameters
DataSet_xy = 'data_balanced_6Slices_1orMore_xy.bin'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz.bin'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz.bin'

xyPlaneInputPath = 'CadLung/INPUT/PLANE/xyPlaneInput.npz'
xzPlaneInputPath = 'CadLung/INPUT/PLANE/xzPlaneInput.npz'
yzPlaneInputPath = 'CadLung/INPUT/PLANE/yzPlaneInput.npz'

volumeInputPath = 'CadLung/INPUT/VOLUME/volumeInput.npz'
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

# Volume Model
volumeModel = NasFcAnn.NasFcAnn(dataPath=volumeInputPath, type='volume',
                              name='volume', regRate=0.001, positiveRegion='n',
                              normalize='zs')

volumeModel.loadData()
volumeModel.doPreProcess()

volumeModel.loadModel()
volumeModel.testModel()
print('\nVolume Model Evaluation:')
volumeModel.evaluate()
###
print(xySliceModel.bestModel.summary())
print(xzSliceModel.bestModel.summary())
print(yzSliceModel.bestModel.summary())
print('Plane')
print(xyPlaneModel.bestModel.summary())
print(xzPlaneModel.bestModel.summary())
print(yzPlaneModel.bestModel.summary())
print('Volume')
print(volumeModel.bestModel.summary())
print('done!')
