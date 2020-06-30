'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: June 16 2020
'''

'''
Author: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program
'''

'''~~~~ CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible. Read the coding standarts and best practice notes.
~~~~'''

'''~~~~ HELP
#Enter your notes to help to understand your code here



~~~~'''

'''IMPORTS'''
#imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from array import array

'''~~~~ CLASSES ~~~~'''
#Your class definitions

'''~~~~ end of CLASSES ~~~~'''


'''~~~~ FUNCTIONS ~~~~'''
#Your function definitions
#plot_train_history: function to plot model performance
def plot_train_history(history, title, performance):
    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #options include: loss, val_loss, accuracy, val_accuracy

    epochs = range(len(loss))   #number of epochs

    #plotting training and validation loss
    axs[0].plot(epochs, loss, 'blue', label='Training loss')
    axs[0].plot(epochs, val_loss, 'red', label='Validation loss')
    axs[0].legend()
    #

    #plotting test performance metrics
    x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']    #creating performance labels
    x_pos = [i for i, _ in enumerate(x)]

    axs[1].bar(x_pos, performance, color=('green', 'red', 'blue', 'orange'))
    axs[1].set_ylim([0, 1])                                         #setting performance score range  
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(x)                                       #setting performance labels
    #

    plt.savefig('CadLung/EXP1/OUTPUT/performance.png')              #Exporting performance chart

    plt.show()
    #####

#save_csv_bin: function to convert dict to csv and bin file
def save_csv_bin(path_csv, path_bin, data_dict):
    #saves as csv file
    with open(path_csv, 'w') as f:
        for key in data_dict.keys():
            f.write("%s,%s\n"%(key,data_dict[key]))
    
    #saves as bin file
    with open(path_bin, 'wb') as file:
        for i in data_dict.values():
            if type(i) == type(str()):
                file.write(i.encode('ascii'))   #converts strings to ascii and writes to bin
            else:
                file.write(i)                   #writes to bin
#####

#delete_old_data: function to delete data from previous runs
def delete_old_data(PATH_EXP):
    if (os.path.exists(PATH_EXP+'INPUT/sampleID.csv')):
        os.remove(PATH_EXP+'INPUT/sampleID.csv')
    if (os.path.exists(PATH_EXP+'INPUT/sampleID.bin')):
        os.remove(PATH_EXP+'INPUT/sampleID.bin')

    if (os.path.exists(PATH_EXP+'MODEL/model.h5')):
        os.remove(PATH_EXP+'MODEL/model.h5')
    if (os.path.exists(PATH_EXP+'MODEL/parameter.csv')):
        os.remove(PATH_EXP+'MODEL/parameter.csv')
    if (os.path.exists(PATH_EXP+'MODEL/parameter.bin')):
        os.remove(PATH_EXP+'MODEL/parameter.bin')

    if (os.path.exists(PATH_EXP+'OUTPUT/performance.csv')):
        os.remove(PATH_EXP+'OUTPUT/performance.csv')
    if (os.path.exists(PATH_EXP+'OUTPUT/performance.bin')):
        os.remove(PATH_EXP+'OUTPUT/performance.bin')
    if (os.path.exists(PATH_EXP+'OUTPUT/performance.png')):
        os.remove(PATH_EXP+'OUTPUT/performance.png')
    if (os.path.exists(PATH_EXP+'OUTPUT/testOutput.bin')):
        os.remove(PATH_EXP+'OUTPUT/testOutput.bin')
#####

'''~~~~ end of FUNCTIONS ~~~~'''


'''~~~~ PARAMETERS ~~~~'''
###
#Use tuple simple or multi-dimensional-like for more detail settings. Add aditional constants for detail of each parameters to export.
PATH_VOI = 'CadLung/INPUT/Voi_Data/'
PATH_EXP = 'CadLung/EXP1/'

delete_old_data(PATH_EXP)   #using function to delete old data before we create new data

#Experiment ID
EXP_ID = 'Exp_1'

#Network Architecture Parameters
#NUM_NODE = (5, 4, 2): Two hidden layers with 5 and 4 nodes, respectively. The output layer has 2 nodes.

NUM_NODE = '5 4 2'

#Model Parameters
#ACT_FUN = ('sigmoid', 'selu', 'relu'): The activation functions in the first and the second hidden layers are Sigmoid and SeLU,
# respectively. At the output layer, it is ReLU.

ACT_FUN = 'relu'                 #Activation function.
INITZR = 'RandomNormal'           #Initializer
LOS_FUN = 'mean_squared_error'
OPTMZR = 'SGD'
#add more as needed

#creating parameter data
parameter_dict = {'path_voi':PATH_VOI, 'exp_id':EXP_ID, 'num_node':NUM_NODE, 'act_fun':ACT_FUN,\
                 'initializer':INITZR, 'loss_fun':LOS_FUN, 'optimizer':OPTMZR, 'path_exp':PATH_EXP}

#using function to save parameter data
save_csv_bin(PATH_EXP+'MODEL/parameter.csv', PATH_EXP+'MODEL/parameter.bin', parameter_dict)
#


'''~~~~ MODEL SETUP ~~~~'''
#your code to create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56, 56)),                                         #input layer, flattening images to one-dimensional array
    keras.layers.Dense(5, activation='sigmoid', kernel_initializer='random_normal'),    #hidden layer, sigmoid activation function
    keras.layers.Dense(4, activation='selu', kernel_initializer='random_normal'),       #second hidden layer, selu activation function
    keras.layers.Dense(1, activation='relu', kernel_initializer='random_normal')        #output layer, relu activation function
])

model.compile(optimizer='SGD',                  #optimizer: stochastic gradient descent
              loss='mean_squared_error',        #loss function: mean squared error
              metrics=['accuracy']) 

'''~~~~ LOAD DATA ~~~~'''
#your code to load data. Export ID of each sample as 'train', 'val', 'test' as a csv and binary file sampleID.csv and sampleID.bin
#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy' #front view of lung
DataSet_xz = 'data_balanced_6Slices_1orMore_xz' #top view of lung
DataSet_yz = 'data_balanced_6Slices_1orMore_yz' #side view of lung

#Obtaining data for front view of lung
f2 = open(PATH_VOI + DataSet_xy + '.bin','rb') 
train_set_all_xy = np.load(f2).reshape(2520, 56, 56)    #training data, reshaped 2520 images to 56x56 pixels
train_label_all_xy = np.load(f2)                        #2520 correct labels for training set
test_set_all_xy = np.load(f2).reshape(1092, 56, 56)     #test data, reshaped 1092 images to 56x56 pixels 30%
test_label_all_xy = np.load(f2)                         #1092 correct labels for test set 30%
f2.close()

train_data_xy = train_set_all_xy[:2170]             #creating training data 60%
train_data_label_xy = train_label_all_xy[:2170]     #creating training labels 60%

validation_set_xy = train_set_all_xy[-350:]         #creating validation data 10%
validation_label_xy = train_label_all_xy[-350:]     #creating validation labels 10%

###
#creating sampleID data and saving as csv and bin file
performance_dict = {'train':train_data_xy, 'val':validation_set_xy, 'test':test_set_all_xy}

#use function to save sampleID data to csv and bin file
save_csv_bin(PATH_EXP+'INPUT/sampleID.csv', PATH_EXP+'INPUT/sampleID.bin', performance_dict)
###

'''~~~~ PRE-PROCESS ~~~~'''
#your code to pre-proces data. Export pre-processed data if takes too long to repeat as a binary file dataPreProcess.bin


'''~~~~ TRAINING ~~~~'''
#your code to train the model. Export trained model parameters to load later as model.bin
model_history = model.fit(train_data_xy, train_data_label_xy, batch_size=32, epochs=10, validation_data=(validation_set_xy, validation_label_xy))      
#training model and saving history
#saving history is required to plot training and validation loss

model.save(PATH_EXP+'MODEL/model.h5')                           #Exporting Model as h5 file

'''~~~~ TESTING ~~~~'''
#your code to test the model. Export predictions with sample-IDs as testOutput.bin
model = keras.models.load_model(PATH_EXP+'MODEL/model.h5')      #Importing Model

#predicted test values (1092) 1 nodes per prediction need to round and reshape for labels
test_pred_all_xy = []
for i in model.predict(test_set_all_xy).reshape(1092):
    test_pred_all_xy.append(round(i))

###Exports test output to testOutput.bin
with open(PATH_EXP+'OUTPUT/testOutput.bin', 'wb') as file:
    for i in test_pred_all_xy:
            file.write(bytes(i))                   #writes to bin
###

'''~~~~ EVALUATION ~~~~'''
#your code to evaluate performance of the model. Export performance metrics as csv and binary file performance.csv and performance.bin
f1score = f1_score(test_label_all_xy, test_pred_all_xy, average='weighted')         #Calculate F1-score
print('F1-score = {}'.format(f1score))
aucRoc = roc_auc_score(test_label_all_xy, test_pred_all_xy)                         #Calculate AUC of ROC
print('AUC of ROC: {}'.format(aucRoc))

confMatrix = confusion_matrix(test_label_all_xy, test_pred_all_xy)                  #Calculate Confusion Matrix
print('Confusion Matrix : \n', confMatrix)

sensitivity = confMatrix[0,0]/(confMatrix[0,0]+confMatrix[0,1])                     #Calculate Sensitivity
print('Sensitivity: {}'.format(sensitivity))
specificity = confMatrix[1,1]/(confMatrix[1,0]+confMatrix[1,1])                     #Calculate Specificity
print('Specificity: {}'.format(specificity))

###
#creating performance data and saving as csv and bin file
performance_dict = {'F1-score':f1score, 'AUC of ROC':aucRoc, 'Sensitivity':sensitivity, 'Specificity':specificity}
save_csv_bin(PATH_EXP+'OUTPUT/performance.csv', PATH_EXP+'OUTPUT/performance.bin', performance_dict)
###

'''~~~~ VISUALIZE ~~~~'''
#your code to visualize performance metrics. Export charts.
performance = [f1score, aucRoc, sensitivity, specificity]   #creating performance list
#Using plot_train_history function to plot model performance
plot_train_history(model_history, 'Model XY Performance', performance)
#####

print('Done!')
