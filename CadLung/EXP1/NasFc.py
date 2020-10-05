'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: July 9 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
- Read the coding standarts and best practice notes.


HELP
'''
import os
import keras

from keras import callbacks as cB
from keras import regularizers as rg
from keras.layers import Dense, Flatten
from keras.metrics import AUC, FalsePositives, TrueNegatives, Precision, Recall
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
'''~~~~ CLASSES ~~~~'''


'''~~~~ PARAMETERS ~~~~'''
#Paths
PATH_VOI = 'CadLung/INPUT/Voi_Data/'
PATH_EXP = 'CadLung/EXP1/'
checkpoint_filepath = PATH_EXP+'CHKPNT/checkpoint.hdf5'
bestModel_filepath = PATH_EXP+'MODEL/bestModel.h5'
performancePng_filepath = PATH_EXP+'OUTPUT/bestModelPerformance.png'

#Data
POSTNORMALIZE = 'y' #Options: {'y', 'n'}
NORMALIZE = 'z-score' #Options: {Default: 'none', 'range', 'z-score'}

#Model
VAL_SPLIT = 0.15
regRate = 0.001
regFn = rg.l2(regRate)
EPOCHS = 10
BATCH_SIZE = 32
METRICS = ['accuracy', AUC(), Recall(), Precision(), FalsePositives(), TrueNegatives()]
LOSS_F = 'binary_crossentropy'
INITZR = 'random_normal'
OPT_M = 'adam'
modelChkPnt_cBk = cB.ModelCheckpoint(filepath=checkpoint_filepath,
                                     save_weights_only=True,
                                     monitor='val_accuracy',
                                     mode='max',
                                     save_best_only=True)
clBacks = [modelChkPnt_cBk]

#Network Architecture
MAX_NUM_NODES = (None, 6, 6, 6, 1) #first layer, hidden layers, and output layer. #hidden nodes > 1.
lenMaxNumHidenLayer = len(MAX_NUM_NODES) - 2    #3
ACT_FUN = (None, 'relu', 'relu', 'relu', 'sigmoid') #best activation functions for binary classification

#Export Parameters into parameter.tf
''' COMPLETE THE CODE '''


'''~~~~ FUNCTIONS ~~~~'''
def rangeNormalize(data, lower, upper): #lower, upper = range
    """function to range normalize data"""
    scaler = MinMaxScaler(feature_range=(lower, upper))
    normalized = scaler.fit_transform(data)
    return normalized
#####

def positiveNormalize(data):
    """function to move data to the positive region"""
    for i in range(len(data)):
        dataMin = min(data[i])
        if dataMin < 0.0001:
            scal = 0.0001-dataMin
            data[i] = data[i] + scal    #shifting elements to make minimum 0.0001
    return data
#####

def plot_performance_metrics(performance):
    '''plotting test performance metrics bar chart'''
    x = ['Accuracy', 'AUC of ROC']    #creating performance labels
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance, color=('blue', 'red'))
    plt.ylim([0, 1])     #setting performance score range
    plt.xticks(x_pos, x)
    plt.title('Best Model Performance Metrics')

    plt.savefig(performancePng_filepath)
    plt.show()
#####


'''~~~~ LOAD DATA ~~~~'''
#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
#DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
#DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

with open(PATH_VOI + DataSet_xy + '.bin', 'rb') as f2:
    train_set_all_xy = np.load(f2)
    train_label_all_xy = np.load(f2)
    test_set_all_xy = np.load(f2)
    test_label_all_xy = np.load(f2)
#

#Export ID of each sample as 'train', 'val', 'test' into sampleID.tf
''' COMPLETE THE CODE '''



'''~~~~ PRE-PROCESS ~~~~'''
#your code to pre-proces data.
#Export pre-processed data if takes too long to create as a binary file dataPreProcess.bin
#Normalize your data
if POSTNORMALIZE == 'y':
    train_set_all_xy = positiveNormalize(train_set_all_xy)
    test_set_all_xy = positiveNormalize(test_set_all_xy)

if NORMALIZE[0] == 'r':
    train_set_all_xy = rangeNormalize(train_set_all_xy, 0, 1)
    test_set_all_xy = rangeNormalize(test_set_all_xy, 0, 1)
elif NORMALIZE[0] == 'z':
    train_set_all_xy = stats.zscore(train_set_all_xy)
    test_set_all_xy = stats.zscore(test_set_all_xy)
#

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

if os.path.exists(bestModel_filepath):
    #compile=False to load AUC() properly
    bestModel = keras.models.load_model(bestModel_filepath, custom_objects={'AUC': AUC()}, compile=False) #Loading best Model
    bestModel.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy', AUC()])#METRICS)
else:
    bestModel = keras.Sequential([
        Flatten(),
        Dense(1, activation=ACT_FUN[1], kernel_initializer=INITZR, kernel_regularizer=regFn), #1st hidden layer
        Dense(1, activation=ACT_FUN[-1], kernel_initializer=INITZR, kernel_regularizer=regFn) #output layer
    ])

    bestModel.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy', AUC()])#METRICS)

    modelFit = bestModel.fit(train_set_all_xy, train_label_all_xy,
                             epochs=EPOCHS, verbose=1, validation_split=VAL_SPLIT)

    bestModel.save(bestModel_filepath)  #saving best model

initLoss, initAcc, initAUC = bestModel.evaluate(test_set_all_xy, test_label_all_xy, verbose=0)

print('Initial Model Structure:')
print(bestModel.summary())  #Printing model structure
print('Initial Accuracy: {}'.format(initAcc))   #Printing initial accuracy
print('Initial AUC: {}'.format(initAUC))   #Printing initial accuracy

numNodeLastHidden = np.zeros(lenMaxNumHidenLayer + 1) #1st one is for input. number of nodes added to the current layer
#[0, 0, 0, 0]
#Searching the best network architecture
for hL in range(1, lenMaxNumHidenLayer+1):  #Hidden Layer Loop 3 layers
    for j in range(1, MAX_NUM_NODES[hL]+1):   #Node loop   (2 to 6), 3 times
        numNodeLastHidden[hL] += 1  #A new node added to the current layer
        #Re-create the temp model with a new node at the layer
        modelTmp = keras.Sequential()   #initialize temporary model
        modelTmp.add(Flatten())         #Input layer

        for iL in range(1, hL+1):             #Adds number of hidden layers
            modelTmp.add(Dense(int(numNodeLastHidden[iL]), activation=ACT_FUN[hL], \
                           kernel_initializer=INITZR, kernel_regularizer=regFn))

        #output layer
        modelTmp.add(Dense(1, activation=ACT_FUN[-1], kernel_initializer=INITZR, kernel_regularizer=regFn))


        modelTmp.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy', AUC()]) #, metrics=METRICS)
        modelFitTmp = modelTmp.fit(train_set_all_xy, train_label_all_xy, batch_size=BATCH_SIZE,
                                   epochs=EPOCHS, verbose=0, callbacks=clBacks,
                                   validation_split=VAL_SPLIT)
        print(modelTmp.summary())
        #After pulling out the best weights and the corresponding model "modelFitTmp",
        modelTmp.load_weights(checkpoint_filepath, by_name=True, skip_mismatch=True)    #loading test weights
        #modelTmp test evaluation
        tmpLoss, tmpAcc, tmpAUC = modelTmp.evaluate(test_set_all_xy, test_label_all_xy, verbose=0)

        #compare against the last "bestAcc"
        if tmpAcc > bestAcc:
            print('Test Accuracy: {}'.format(tmpAcc))
            #update the best model and continue adding a node to this layer
            bestAcc = tmpAcc
            bestModel = modelTmp
            del modelTmp    #WHY ?
        else:   #adding a new node did not improve the performance.
            if numNodeLastHidden[hL] != 1:
                numNodeLastHidden[hL] -= 1  #going back to best number of nodes
            #Stop adding a new node to this layer
            break
        #
    #for j
#for hL

'''#~~~~ TESTING ~~~~'''
bestModel.load_weights(checkpoint_filepath, by_name=True, skip_mismatch=True)    #loading test weights
#bestModel evaluation
testLoss, testAcc, testAUC = bestModel.evaluate(test_set_all_xy, test_label_all_xy, verbose=0)

print('\nTest Model Structure:')
print(bestModel.summary())  #Printing new model structure
print('Test Accuracy: {}'.format(testAcc))
print('Test AUC: {}'.format(testAUC))

#if model performed better than initial, save new one
if testAcc > initAcc:
    print('Test Model Performed Better\n')
    bestModel.save(bestModel_filepath)  #saving best model
else:
    print('Test Model Performed Worse\n')

#Export predictions with sample-IDs into testOutput.tf
''' #COMPLETE THE CODE '''



'''#~~~~ EVALUATION ~~~~'''
#Evaluate performance of the model
#compile=False to load AUC() properly
bestModel = keras.models.load_model(bestModel_filepath, custom_objects={'AUC': AUC()}, compile=False) #Loading best Model
bestModel.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy', AUC()])#METRICS)

bestLoss, bestAcc, bestAUC = bestModel.evaluate(test_set_all_xy, test_label_all_xy, verbose=0)

print('Best Model Evaluation:')
print('Accuracy: {}'.format(bestAcc))
print('AUC of ROC: {}'.format(bestAUC))


#Export performance metrics into performance.tf
''' #COMPLETE THE CODE '''


'''~~~~ VISUALIZE ~~~~'''
#Visualize performance metrics
performance = [bestAcc, bestAUC]
plot_performance_metrics(performance)


print('Done!')
