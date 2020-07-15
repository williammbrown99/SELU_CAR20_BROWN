'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: July 9 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible. Read the coding standarts and best practice notes.


HELP


'''

'''~~~~ IMPORTS ~~~~'''
#import os
#import keras as ks
#from keras import backend as k
import keras

from keras import callbacks as cB
from keras import models as kModels
from keras.layers import Activation, Dense, Flatten, Input
from keras.metrics import AUC, FalsePositives, TrueNegatives, Precision, Recall
from keras.models import Sequential
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''~~~~ CLASSES ~~~~'''




'''~~~~ FUNCTIONS ~~~~'''
def rangeNormalize(data, lower, upper): #lower, upper = range
    """function to range normalize data"""
    scaler = MinMaxScaler(feature_range=(lower, upper))
    normalized = scaler.fit_transform(data)
    return normalized
#####

'''~~~~ PARAMETERS ~~~~'''
#Paths
PATH_VOI = 'CadLung/INPUT/Voi_Data/'
PATH_EXP = 'CadLung/EXP1/'

#Data
NORMALIZE = 'z-score' #Options: {Default: 'none', 'range', 'z-score'}

#Model
VAL_SPLIT = 0.15
EPOCHS = 100
BATCH_SIZE = 32
METRICS=['accuracy', AUC(), Recall(), Precision(), FalsePositives(), TrueNegatives()]
LOSS_F = 'mean_squared_error'
OPT_M = 'SGD'
checkpoint_filepath = PATH_EXP+'CHKPNT/bestWeights.hdf5'
modelChkPnt_cBk = cB.ModelCheckpoint(filepath=checkpoint_filepath,
                                               save_weights_only=True,
                                               monitor='val_accuracy',
                                               mode='max',
                                               save_best_only=True)
clBacks = [modelChkPnt_cBk]

#Network Architecture
MAX_NUM_NODES = (None,6,6,6,1)   #first layer, hidden layers, and output layer. #hidden nodes > 1.
lenMaxNumHidenLayer = len(MAX_NUM_NODES) - 2    #3
ACT_FUN = (None,'sigmoid','sigmoid','sigmoid','relu')

#Export Parameters into parameter.tf
''' COMPLETE THE CODE '''



'''~~~~ LOAD DATA ~~~~'''
#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
#DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
#DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

with open(PATH_VOI + DataSet_xy + '.bin','rb') as f2:
    train_set_all_xy = np.load(f2)
    train_label_all_xy = np.load(f2)
    test_set_all_xy = np.load(f2)
    test_label_all_xy = np.load(f2)
#

#Export ID of each sample as 'train', 'val', 'test' into sampleID.tf
''' COMPLETE THE CODE '''



'''~~~~ PRE-PROCESS ~~~~'''
#Normalize your data 
if NORMALIZE[0] == 'r':
    train_set_all_xy = rangeNormalize(train_set_all_xy, 0, 1)
    test_set_all_xy = rangeNormalize(test_set_all_xy, 0, 1)
elif NORMALIZE[0] == 'z':
    train_set_all_xy = stats.zscore(train_set_all_xy)
    test_set_all_xy = stats.zscore(test_set_all_xy)
#
#your code to pre-proces data. Export pre-processed data if takes too long to create as a binary file dataPreProcess.bin
#Dim1: Batch, (Dim2,Dim3): Flattened images
train_set_all_xy = np.reshape(train_set_all_xy, (train_set_all_xy.shape[0], 1, train_set_all_xy.shape[1]))
#Dim1: Batch, (Dim2,Dim3): Flattened images
test_set_all_xy = np.reshape(test_set_all_xy, (test_set_all_xy.shape[0], 1, test_set_all_xy.shape[1]))
#train_label_all_xy = keras.utils.to_categorical(train_label_all_xy)

#Export pre-processed into data dataPreProcess.tf if takes too long to create 
''' COMPLETE THE CODE '''


'''~~~~ MODEL SETUP & TRAINING  ~~~~'''
#Initial Training
bestAcc = 0 #best accuracy score

bestModel = keras.Sequential([
    Flatten(),
    Dense(1, activation=ACT_FUN[1], kernel_initializer='random_normal'), #1st hidden layer
    Dense(1, activation=ACT_FUN[-1], kernel_initializer='random_normal') #output layer
])

bestModel.compile(loss=LOSS_F, optimizer=OPT_M, metrics=METRICS)

#modelFit = bestModel.fit(train_set_all_xy, train_label_all_xy, epochs=EPOCHS,verbose=0,validation_split = VAL_SPLIT)

numNodeLastHidden = np.zeros(lenMaxNumHidenLayer + 1) #[0 0 0 0] 1st one is for input. number of nodes added to the current layer

#Searching the best network architecture
for hL in range(1,lenMaxNumHidenLayer+1):  #Hidden Layer Loop (1 to 4)
    for j in range(2,MAX_NUM_NODES[hL]):   #Node loop   (2 to 6), 3 times
        numNodeLastHidden[hL] += 1  #A new node added to the current layer [0 0 0]
        #Re-create the temp model with a new node at the layer
        modelTmp = keras.Sequential()   #initialize temporary model
        modelTmp.add(Flatten())         #Input layer

        for i in range(hL):             #Adds number of hidden layers
            modelTmp.add(Dense(int(numNodeLastHidden[hL]), activation=ACT_FUN[hL], \
                           kernel_initializer='random_normal'))  #2 nodes, only recieves input from input layer
        
        #output layer
        modelTmp.add(Dense(1, activation=ACT_FUN[-1], kernel_initializer='random_normal'))


        modelTmp.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy']) #, metrics=METRICS)
        modelFitTmp = modelTmp.fit(train_set_all_xy, train_label_all_xy, batch_size=BATCH_SIZE,\
                                   epochs=EPOCHS, verbose=0, callbacks = clBacks,\
                                   validation_split=VAL_SPLIT)

        #Get the best weights
        #!!!!!! THIS SECTION NEEDS TO BE COMPLETED   !!!!!! 
        #Write the code to get the best set of weights "modelFitTmp" out of all epochs

        #After pulling out the best weights and the corresponding model "modelFitTmp", compare against the last "bestAcc"
        if max(modelFitTmp.history['val_accuracy']) >= bestAcc:   #update the best model and continue adding a node to this layer
            bestAcc = max(modelFitTmp.history['val_accuracy'])
            bestModel = modelTmp
            del modelTmp    #WHY ?
        else:   #adding a new node did not improve the performance. Stop adding a new node to this layer
            break
        #
    #for j
#for hL

#Export trained model parameters into model.tf to load later
''' #COMPLETE THE CODE '''



'''#~~~~ TESTING ~~~~'''
''' COMPLETE THE CODE '''
bestModel.load_weights(checkpoint_filepath)

test_pred = []
for i in bestModel.predict(test_set_all_xy).reshape(test_label_all_xy.shape[0]):  #reshape from (1092, 1) to (1092)
    test_pred.append(round(i))
testAcc = accuracy_score(test_label_all_xy, test_pred)

print('Test Accuracy: {}'.format(testAcc))
print(bestModel.summary())


#Export predictions with sample-IDs into testOutput.tf
''' #COMPLETE THE CODE '''


'''#~~~~ EVALUATION ~~~~'''
#Evaluate performance of the model
''' #COMPLETE THE CODE '''



#Export performance metrics into performance.tf
''' #COMPLETE THE CODE '''


'''~~~~ VISUALIZE ~~~~'''
#Visualize performance metrics
''' #COMPLETE THE CODE '''

#Export charts
''' #COMPLETE THE CODE '''


print('Done!')
