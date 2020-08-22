'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: August 21 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
- Read the coding standarts and best practice notes.
~~~~ HELP  ~~~~
- Enter your notes to help to understand your code here
'''
import NasFcAnn
import keras
import struct

#Parameters
paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT/')

#Read XY Slice Model Predictions
with open(paths[1]+'OUTPUT/xySliceTestPredictions.tf', mode='r') as file:
    xySlicePred = file.readlines()

for i in range(0, len(xySlicePred)):
    xySlicePred[i] = float(xySlicePred[i])

print(xySlicePred)
print(len(xySlicePred))

print('done!')
