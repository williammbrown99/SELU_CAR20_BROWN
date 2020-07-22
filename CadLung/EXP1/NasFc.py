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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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
NORMALIZE = 'z-score' #Options: {Default: 'none', 'range', 'z-score'}

#Model
VAL_SPLIT = 0.15
regRate = 0.001
regFn = rg.l2(regRate)
EPOCHS = 100
BATCH_SIZE = 32
METRICS = ['accuracy', AUC(), Recall(), Precision(), FalsePositives(), TrueNegatives()]
LOSS_F = 'mean_squared_error'
OPT_M = 'SGD'
modelChkPnt_cBk = cB.ModelCheckpoint(filepath=checkpoint_filepath,
                                     save_weights_only=True,
                                     monitor='val_accuracy',
                                     mode='max',
                                     save_best_only=True)
clBacks = [modelChkPnt_cBk]

#Network Architecture
MAX_NUM_NODES = (None, 6, 6, 6, 1) #first layer, hidden layers, and output layer. #hidden nodes > 1.
lenMaxNumHidenLayer = len(MAX_NUM_NODES) - 2    #3
ACT_FUN = (None, 'sigmoid', 'sigmoid', 'sigmoid', 'relu')

#Export Parameters into parameter.tf
''' COMPLETE THE CODE '''


'''~~~~ FUNCTIONS ~~~~'''
def rangeNormalize(data, lower, upper): #lower, upper = range
    """function to range normalize data"""
    scaler = MinMaxScaler(feature_range=(lower, upper))
    normalized = scaler.fit_transform(data)
    return normalized
#####

def plot_performance_metrics(performance):
    '''plotting test performance metrics bar chart'''
    x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']    #creating performance labels
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, performance, color=('green', 'red', 'blue', 'yellow'))
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
    bestModel = keras.models.load_model(bestModel_filepath) #Loading best Model
else:
    bestModel = keras.Sequential([
        Flatten(),
        Dense(1, activation=ACT_FUN[1], kernel_initializer='random_normal'),#, kernel_regularizer = regFn), #1st hidden layer
        Dense(1, activation=ACT_FUN[-1], kernel_initializer='random_normal')#, kernel_regularizer = regFn) #output layer
    ])

    bestModel.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy'])#METRICS)

    modelFit = bestModel.fit(train_set_all_xy, train_label_all_xy,
                             epochs=EPOCHS, verbose=0, validation_split=VAL_SPLIT)

#making initial prediction on test set
init_pred = []
#reshape from (1092, 1) to (1092)
for i in bestModel.predict(test_set_all_xy).reshape(test_label_all_xy.shape[0]):
    init_pred.append(round(i))
initAcc = accuracy_score(test_label_all_xy, init_pred)

print(bestModel.summary())  #Printing model structure
print('Initial Accuracy: {}'.format(initAcc))   #Printing initial accuracy

#[0 0 0 0] 1st one is for input. number of nodes added to the current layer
#numNodeLastHidden = np.zeros(lenMaxNumHidenLayer + 1)
numNodeLastHidden = [1, 1, 1, 1]    #hidden node > 1

#Searching the best network architecture
for hL in range(1, lenMaxNumHidenLayer+1):  #Hidden Layer Loop (1 to 4)
    for j in range(2, MAX_NUM_NODES[hL]+1):   #Node loop   (2 to 6), 3 times
        numNodeLastHidden[hL] += 1  #A new node added to the current layer [0 0 0]
        #Re-create the temp model with a new node at the layer
        modelTmp = keras.Sequential()   #initialize temporary model
        modelTmp.add(Flatten())         #Input layer

        for iL in range(1, hL+1):             #Adds number of hidden layers
            modelTmp.add(Dense(int(numNodeLastHidden[iL]), activation=ACT_FUN[hL], \
                           kernel_initializer='random_normal'))#, kernel_regularizer = regFn))

        #output layer
        modelTmp.add(Dense(1, activation=ACT_FUN[-1], kernel_initializer='random_normal'))#, kernel_regularizer = regFn))


        modelTmp.compile(loss=LOSS_F, optimizer=OPT_M, metrics=['accuracy']) #, metrics=METRICS)
        modelFitTmp = modelTmp.fit(train_set_all_xy, train_label_all_xy, batch_size=BATCH_SIZE,
                                   epochs=EPOCHS, verbose=0, callbacks=clBacks,
                                   validation_split=VAL_SPLIT)

        #After pulling out the best weights and the corresponding model "modelFitTmp",
        #compare against the last "bestAcc"
        if max(modelFitTmp.history['val_accuracy']) > bestAcc:
            print(max(modelFitTmp.history['val_accuracy']))
            #update the best model and continue adding a node to this layer
            bestAcc = max(modelFitTmp.history['val_accuracy'])
            bestModel = modelTmp
            del modelTmp    #WHY ?
        else:   #adding a new node did not improve the performance.
            #Stop adding a new node to this layer
            break
        #
    #for j
#for hL

'''#~~~~ TESTING ~~~~'''
bestModel.load_weights(checkpoint_filepath)    #loading test weights

#making test prediction
test_pred = []
#reshape from (1092, 1) to (1092)
for i in bestModel.predict(test_set_all_xy).reshape(test_label_all_xy.shape[0]):
    test_pred.append(round(i))
testAcc = accuracy_score(test_label_all_xy, test_pred)  #test accuracy

print(bestModel.summary())  #Printing new model structure
print('Test Accuracy: {}'.format(testAcc))

#if model performed better than initial, save new one
if testAcc > initAcc:
    bestModel.save(bestModel_filepath)  #saving best model

#Export predictions with sample-IDs into testOutput.tf
''' #COMPLETE THE CODE '''



'''#~~~~ EVALUATION ~~~~'''
#Evaluate performance of the model
bestModel = keras.models.load_model(bestModel_filepath) #Loading best Model

#finding best predictions
best_pred = []
#reshape from (1092, 1) to (1092)
for i in bestModel.predict(test_set_all_xy).reshape(test_label_all_xy.shape[0]):
    best_pred.append(round(i))

#Calculate F1-score
f1score = f1_score(test_label_all_xy, best_pred, average='weighted')
print('F1-score = {}'.format(f1score))
#Calculate AUC of ROC
aucRoc = roc_auc_score(test_label_all_xy, best_pred)
print('AUC of ROC: {}'.format(aucRoc))
#Calculate Confusion Matrix
confMatrix = confusion_matrix(test_label_all_xy, best_pred)
print('Confusion Matrix : \n', confMatrix)
#Calculate Sensitivity
sensitivity = confMatrix[0, 0]/(confMatrix[0, 0]+confMatrix[0, 1])
print('Sensitivity: {}'.format(sensitivity))
#Calculate Specificity
specificity = confMatrix[1, 1]/(confMatrix[1, 0]+confMatrix[1, 1])
print('Specificity: {}'.format(specificity))


#Export performance metrics into performance.tf
''' #COMPLETE THE CODE '''


'''~~~~ VISUALIZE ~~~~'''
#Visualize performance metrics
performance = [f1score, aucRoc, sensitivity, specificity]
plot_performance_metrics(performance)


print('Done!')
