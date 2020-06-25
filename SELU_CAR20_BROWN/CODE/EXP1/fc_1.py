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

from tensorflow import keras

'''~~~~ CLASSES ~~~~'''
#Your class definitions

'''~~~~ end of CLASSES ~~~~'''


'''~~~~ FUNCTIONS ~~~~'''
#Your function definitions

'''~~~~ end of FUNCTIONS ~~~~'''



'''~~~~ PARAMETERS ~~~~'''
###
#Use tuple simple or multi-dimensional-like for more detail settings. Add aditional constants for detail of each parameters to export.
PATH_VOI = 'SELU_CAR20_BROWN/CODE/INPUT/Voi_Data/'

#Experiment ID
EXP_ID = 'Exp_1'

#Network Architecture Parameters
#NUM_NODE = (5, 4, 2): Two hidden layers with 5 and 4 nodes, respectively. The output layer has 2 nodes.

NUM_NODE = (1,1)

#Model Parameters
#ACT_FUN = ('sigmoid', 'selu', 'relu'): The activation functions in the first and the second hidden layers are Sigmoid and SeLU,
# respectively. At the output layer, it is ReLU.

ACT_FUN = ('relu') #Activation function.
INITZR = ('RandomNormal') #Initializer
LOS_FUN = ('mean_squared_error')
OPTMZR = ('SGD')
#add more as needed

#Export Parameters as a csv and binary file parameter.csv and parameter.bin


'''~~~~ MODEL SETUP ~~~~'''
#your code to create the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56, 56)), #flattening images
    keras.layers.Dense(5, activation='sigmoid', kernel_initializer='random_normal'),    #hidden layer, sigmoid activation function
    keras.layers.Dense(4, activation='selu', kernel_initializer='random_normal'),       #second hidden layer, selu activation function
    keras.layers.Dense(2, activation='relu', kernel_initializer='random_normal')        #output layer, relu activation function
])

model.compile(optimizer='SGD',                  #optimizer, Stochastic gradient descent
              loss='mean_squared_error',        #loss function, mean squared error
              metrics=['accuracy']) 

'''~~~~ LOAD DATA ~~~~'''
#your code to load data. Export ID of each sample as 'train', 'val', 'test' as a csv and binary file sampleID.csv and sampleID.bin
#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

#Obtaining data
f2 = open(PATH_VOI + DataSet_xy + '.bin','rb') 
train_set_all_xy = np.load(f2)          #training parameter
train_label_all_xy = np.load(f2)        #training labels
test_set_all_xy = np.load(f2)
test_label_all_xy = np.load(f2)
f2.close()

'''~~~~ PRE-PROCESS ~~~~'''
#your code to pre-proces data. Export pre-processed data if takes too long to repeat as a binary file dataPreProcess.bin  


'''~~~~ TRAINING ~~~~'''
#your code to train the model. Export trained model parameters to load later as model.bin
model.fit(train_set_all_xy, train_label_all_xy, epochs=10)  #training model


'''~~~~ TESTING ~~~~'''
#your code to test the model. Export predictions with sample-IDs as testOutput.bin



'''~~~~ EVALUATION ~~~~'''
#your code to evaluate performance of the model. Export performance metrics as csv and binary file performance.csv and performance.bin


'''~~~~ VISUALIZE ~~~~'''
#your code to visualize performance metrics. Export charts.


print('Done!')
