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

from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

'''~~~~ CLASSES ~~~~'''
#Your class definitions

'''~~~~ end of CLASSES ~~~~'''


'''~~~~ FUNCTIONS ~~~~'''
#Your function definitions

'''~~~~ end of FUNCTIONS ~~~~'''



'''~~~~ PARAMETERS ~~~~'''
###
#Use tuple simple or multi-dimensional-like for more detail settings. Add aditional constants for detail of each parameters to export.
PATH_VOI = 'CadLung/INPUT/Voi_Data/'

#Experiment ID
EXP_ID = 'Exp_1'

#Network Architecture Parameters
#NUM_NODE = (5, 4, 2): Two hidden layers with 5 and 4 nodes, respectively. The output layer has 2 nodes.

NUM_NODE = (1,1)

#Model Parameters
#ACT_FUN = ('sigmoid', 'selu', 'relu'): The activation functions in the first and the second hidden layers are Sigmoid and SeLU,
# respectively. At the output layer, it is ReLU.

ACT_FUN = ('relu')          #Activation function.
INITZR = ('RandomNormal')   #Initializer
LOS_FUN = ('mean_squared_error')
OPTMZR = ('SGD')
#add more as needed

#Export Parameters as a csv and binary file parameter.csv and parameter.bin


'''~~~~ MODEL SETUP ~~~~'''
#your code to create the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56, 56)),                                         #flattening images
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
train_set_all_xy = np.load(f2).reshape(2520, 56, 56)    #training data, reshaped images to 56x56 pixels
train_label_all_xy = np.load(f2)                        #training labels
test_set_all_xy = np.load(f2).reshape(1092, 56, 56)     #test data, reshaped images 56x56 pixels
test_label_all_xy = np.load(f2)                         #test labels
f2.close()

'''~~~~ PRE-PROCESS ~~~~'''
#your code to pre-proces data. Export pre-processed data if takes too long to repeat as a binary file dataPreProcess.bin  


'''~~~~ TRAINING ~~~~'''
#your code to train the model. Export trained model parameters to load later as model.bin
model.fit(train_set_all_xy, train_label_all_xy, epochs=10)  #training model


'''~~~~ TESTING ~~~~'''
#your code to test the model. Export predictions with sample-IDs as testOutput.bin
predictions = model.predict(test_set_all_xy)            #predicted test values (1092, 2) 2 nodes per prediction

test_pred_all_xy = []                                   #creating predictions list
for i in range(len(test_label_all_xy)):
    test_pred_all_xy.append(np.argmax(predictions[i]))  #making prediction using the highest valued node

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

'''~~~~ VISUALIZE ~~~~'''
#your code to visualize performance metrics. Export charts.
plt.style.use('ggplot')

x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']
energy = [f1score, aucRoc, sensitivity, specificity]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("Performance Metric")
plt.ylabel("Score")
plt.title("Performance Metrics of Model")

plt.xticks(x_pos, x)

plt.show()


print('Done!')
