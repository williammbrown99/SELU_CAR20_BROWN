
'''
SELU - CAR System - LBRN 2020 Virtual Summer Program
Omer M. Soysal
June 11, 2020
'''
import matplotlib.pyplot as plt
import numpy as np

#
import os
print(os.getcwd()) #prints current path
#

PATH_VOI = 'CadLung/INPUT/Voi_Data/'

#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

f2 = open(PATH_VOI + DataSet_xy + '.bin','rb') 
train_set_all_xy = np.load(f2).reshape(2520, 56, 56)    #reshapes 2520 images to 56x56
train_label_all_xy = np.load(f2)                        #2520 correct labels for training set
test_set_all_xy = np.load(f2).reshape(1092, 56, 56)     #reshapes 1092 images to 56x56
test_label_all_xy = np.load(f2)                         #1092 correct labels for test set
f2.close()

###Views first 25 images along with correct label###
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_set_all_xy[i], cmap=plt.cm.binary)
    plt.xlabel(train_label_all_xy[i])
plt.show()
#####

###Views each image individually###
for i in range(len(train_label_all_xy)):
    plt.imshow(train_set_all_xy[i], cmap=plt.cm.bone)
    plt.show()
    print(i)
    plt.pause(0)
#####
