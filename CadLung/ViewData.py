
'''
SELU - CAR System - LBRN 2020 Virtual Summer Program
Omer M. Soysal
June 11, 2020
'''
import matplotlib.pyplot as plt
import numpy as np

#
import os
print(os.getcwd())
#

PATH_VOI = 'CadLung/INPUT/Voi_Data/'

#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

f2 = open(PATH_VOI + DataSet_xy + '.bin','rb') 
train_set_all_xy = np.load(f2)
train_label_all_xy = np.load(f2)
test_set_all_xy = np.load(f2)
test_label_all_xy = np.load(f2)
f2.close()

for i in range(len(train_label_all_xy)):
    plt.imshow(train_set_all_xy[i].reshape(56,56), cmap=plt.cm.bone)
    plt.show()
    print(i)
    plt.pause(0)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_set_all_xy[i].reshape(56,56), cmap=plt.cm.binary)
    plt.xlabel(train_label_all_xy[i])
plt.show()