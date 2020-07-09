"""SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: June 16 2020
Author: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program
~~~~ CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
Read the coding standarts and best practice notes.
~~~~
~~~~ HELP
#Enter your notes to help to understand your code here
~~~~"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from tensorflow import keras
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

'''~~~~ CLASSES ~~~~'''
#Your class definitions

'''~~~~ end of CLASSES ~~~~'''


'''~~~~ FUNCTIONS ~~~~'''
#Your function definitions
def save_csv_bin(path_csv, path_bin, data_dict):
    """function to convert dict to csv and bin file"""
    with open(path_csv, 'w') as f:
        for key in data_dict.keys():
            f.write("%s,%s\n"%(key, data_dict[key]))
    #saves as bin file
    with open(path_bin, 'wb') as file:
        for i in data_dict.values():
            if type(i) == type(str()):
                file.write(i.encode('ascii'))   #converts strings to ascii and writes to bin
            else:
                file.write(i)                   #writes to bin
#####

def delete_old_data():
    """function to delete data from previous runs"""
    if os.path.exists(PATH_EXP+'INPUT/sampleID.csv'):
        os.remove(PATH_EXP+'INPUT/sampleID.csv')
    if os.path.exists(PATH_EXP+'INPUT/sampleID.bin'):
        os.remove(PATH_EXP+'INPUT/sampleID.bin')
    if os.path.exists(PATH_EXP+'INPUT/PROCESSED/dataPreProcess.bin'):
        os.remove(PATH_EXP+'INPUT/PROCESSED/dataPreProcess.bin')

    if os.path.exists(PATH_EXP+'MODEL/model.h5'):
        os.remove(PATH_EXP+'MODEL/model.h5')
    if os.path.exists(PATH_EXP+'MODEL/parameter.csv'):
        os.remove(PATH_EXP+'MODEL/parameter.csv')
    if os.path.exists(PATH_EXP+'MODEL/parameter.bin'):
        os.remove(PATH_EXP+'MODEL/parameter.bin')

    if os.path.exists(PATH_EXP+'OUTPUT/performance.csv'):
        os.remove(PATH_EXP+'OUTPUT/performance.csv')
    if os.path.exists(PATH_EXP+'OUTPUT/performance.bin'):
        os.remove(PATH_EXP+'OUTPUT/performance.bin')
    if os.path.exists(PATH_EXP+'OUTPUT/performance.png'):
        os.remove(PATH_EXP+'OUTPUT/performance.png')
    if os.path.exists(PATH_EXP+'OUTPUT/testOutput.bin'):
        os.remove(PATH_EXP+'OUTPUT/testOutput.bin')
#####

### CUSTOM METRICS ###
def AUC(true, pred):
    """function to calculate AUC of ROC
    https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82689
    We want strictly 1D arrays - cannot have (batch, 1), for instance"""
    true = K.flatten(true)
    pred = K.flatten(pred)

    #total number of elements in this batch
    totalCount = K.shape(true)[0]

    #sorting the prediction values in descending order
    values, indices = tf.nn.top_k(pred, k=totalCount)
    #sorting the ground truth values based on the predictions above
    sortedTrue = K.gather(true, indices)

    #getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

    #the true positive count per threshold
    TPCurve = K.cumsum(sortedTrue)

    #area under the curve
    auc = K.sum(TPCurve * negatives)

    #normalizing the result between 0 and 1
    totalCount = K.cast(totalCount, K.floatx())
    positiveCount = K.sum(true)
    negativeCount = totalCount - positiveCount
    totalArea = positiveCount * negativeCount
    return  auc / totalArea
#####

def sensitivity(y_true, y_pred):
    """function to calculate sensitivity
    https://www.deepideas.net/unbalanced-classes-machine-learning/"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
#####

def specificity(y_true, y_pred):
    """#specificity: function to calculate specificity
    https://www.deepideas.net/unbalanced-classes-machine-learning/"""
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
#####

def f1(y_true, y_pred):
    """function to calculate f1 score
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras"""
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#####
#####

def plot_train_history(history, performance):
    """function to plot model performance"""
    fig, axs = plt.subplots(6)

    #loading model history metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    auc = history.history['AUC']
    val_auc = history.history['val_AUC']
    sensitivity = history.history['sensitivity']
    val_sensitivity = history.history['val_sensitivity']
    specificity = history.history['specificity']
    val_specificity = history.history['val_specificity']
    f1 = history.history['f1']
    val_f1 = history.history['val_f1']

    epochs = range(len(loss))   #number of epochs

    #plotting training and validation loss
    axs[0].plot(epochs, loss, 'yellow', label='Training loss')
    axs[0].plot(epochs, val_loss, 'purple', label='Validation loss')
    axs[0].legend()

    #plotting training and validation AUC
    axs[1].plot(epochs, auc, 'blue', label='Training AUC')
    axs[1].plot(epochs, val_auc, 'orange', label='Validation AUC')
    axs[1].legend()

    #plotting training and validation sensitivity
    axs[2].plot(epochs, sensitivity, 'red', label='Training sensitivity')
    axs[2].plot(epochs, val_sensitivity, 'green', label='Validation sensitivity')
    axs[2].legend()

    #plotting training and validation specificity
    axs[3].plot(epochs, specificity, 'yellow', label='Training specificity')
    axs[3].plot(epochs, val_specificity, 'purple', label='Validation specificity')
    axs[3].legend()

    #plotting training and validation f1 score
    axs[4].plot(epochs, f1, 'blue', label='Training f1')
    axs[4].plot(epochs, val_f1, 'orange', label='Validation f1')
    axs[4].legend()

    #plotting test performance metrics bar chart
    x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']    #creating performance labels
    x_pos = [i for i, _ in enumerate(x)]

    axs[5].bar(x_pos, performance, color=('green', 'red', 'blue', 'yellow'))
    axs[5].set_ylim([0, 1])     #setting performance score range
    axs[5].set_xticks(x_pos)
    axs[5].set_xticklabels(x)   #setting performance labels

    fig.tight_layout(pad=0.25)  #adding more space to figure

    plt.savefig(PATH_EXP+'OUTPUT/performance.png')  #Exporting performance chart

    plt.show()
    #####

def rangeNormalize(data, lower, upper): #lower, upper = range
    """function to range normalize data"""
    scaler = MinMaxScaler(feature_range=(lower, upper))
    nsamples, nx, ny = data.shape
    twoDimData = data.reshape((nsamples, nx*ny)) #reshaping 3d to 2d
    normalized = scaler.fit_transform(twoDimData)
    return normalized.reshape(nsamples, nx, ny) #reshaping back to 3d
#####

def positiveNormalize(data):
    """function to move data to the positive region"""
    nsamples, nx, ny = data.shape
    twoDimData = data.reshape((nsamples, nx*ny)) #reshaping 3d to 2d
    for i in range(len(twoDimData)):
        dataMin = min(twoDimData[i])
        if dataMin < 0.0001:
            scal = 0.0001-dataMin
            twoDimData[i] = twoDimData[i] + scal    #shifting elements to make minimum 0.0001
    postNorm = twoDimData.reshape(nsamples, nx, ny) #reshaping back to 3d
    return postNorm

'''~~~~ end of FUNCTIONS ~~~~'''


'''~~~~ PARAMETERS ~~~~'''
###
#Use tuple simple or multi-dimensional-like for more detail settings.
#Add aditional constants for detail of each parameters to export.
PATH_VOI = 'CadLung/INPUT/Voi_Data/'
PATH_EXP = 'CadLung/EXP1/'

delete_old_data()   #using function to delete old data before we create new data

#Experiment ID
EXP_ID = 'Exp_1'

#Network Architecture Parameters
#NUM_NODE = (5, 4, 2): Two hidden layers with 5 and 4 nodes, respectively.
#The output layer has 2 nodes.

NUM_NODE = '5 4 2'

#Model Parameters
#ACT_FUN = ('sigmoid', 'selu', 'relu'):
#The activation functions in the first and the second hidden layers are Sigmoid and SeLU,
# respectively. At the output layer, it is ReLU.

ACT_FUN = 'relu'                 #Activation function.
INITZR = 'RandomNormal'          #Initializer
LOS_FUN = 'mean_squared_error'
OPTMZR = 'SGD'
#zscore normalization has best results
NORMALIZE = 'z-score' #Options: {Default: 'none', 'range', 'z-score'}
#add more as needed

#creating parameter dict and saving as csv and bin file
parameter_dict = {'path_voi':PATH_VOI, 'exp_id':EXP_ID, 'num_node':NUM_NODE, 'act_fun':ACT_FUN,\
                 'initializer':INITZR, 'loss_fun':LOS_FUN, 'optimizer':OPTMZR, 'path_exp':PATH_EXP,\
                 'normalize':NORMALIZE}
save_csv_bin(PATH_EXP+'MODEL/parameter.csv', PATH_EXP+'MODEL/parameter.bin', parameter_dict)
#

'''~~~~ MODEL SETUP ~~~~'''
#your code to create the model
model = keras.Sequential([
    #input layer, flattening images to one-dimensional array
    keras.layers.Flatten(input_shape=(56, 56)),
    #hidden layer, sigmoid activation function
    keras.layers.Dense(5, activation='sigmoid', kernel_initializer='random_normal'),
    #second hidden layer, selu activation function
    keras.layers.Dense(4, activation='selu', kernel_initializer='random_normal'),
    #1 output layer, relu activation function
    keras.layers.Dense(1, activation='relu', kernel_initializer='random_normal')
])

model.compile(optimizer='SGD',                  #optimizer: stochastic gradient descent
              loss='mean_squared_error',        #loss function: mean squared error
              metrics=[AUC, sensitivity, specificity, f1]) #loading custom metrics from functions

'''~~~~ LOAD DATA ~~~~'''
#your code to load data.
#Export ID of each sample as 'train', 'val', 'test' as a csv and binary file
#sampleID.csv and sampleID.bin
#VOI Data
DataSet_xy = 'data_balanced_6Slices_1orMore_xy' #front view of lung
DataSet_xz = 'data_balanced_6Slices_1orMore_xz' #top view of lung
DataSet_yz = 'data_balanced_6Slices_1orMore_yz' #side view of lung

#Obtaining data for front view of lung
f2 = open(PATH_VOI + DataSet_xy + '.bin', 'rb')
#training data, reshaped 2520 images to 56x56 pixels
train_set_all_xy = np.load(f2).reshape(2520, 56, 56)
#2520 correct labels for training set
train_label_all_xy = np.load(f2)
#2520 correct labels for training set
test_set_all_xy = np.load(f2).reshape(1092, 56, 56)
#1092 correct labels for test set 30%
test_label_all_xy = np.load(f2)
f2.close()

###
#creating sampleID dict and saving as csv and bin file
performance_dict = {'train':train_set_all_xy[:2170],\
                    'val':train_set_all_xy[-350:],\
                    'test':test_set_all_xy}
save_csv_bin(PATH_EXP+'INPUT/sampleID.csv', PATH_EXP+'INPUT/sampleID.bin', performance_dict)
###

'''~~~~ PRE-PROCESS ~~~~'''
#your code to pre-proces data.
#Export pre-processed data if takes too long to repeat as a binary file dataPreProcess.bin
#moving to the positive region of the feature space.
#Minimum value will be greater than or equal to "0.0001".
train_set_all_xy = positiveNormalize(train_set_all_xy)
test_set_all_xy = positiveNormalize(test_set_all_xy)

#Normalize your data
if NORMALIZE[0] == 'r':                                         #range normalizing between 0 and 1
    train_set_all_xy = rangeNormalize(train_set_all_xy, 0, 1)    #(data, min, max)
    test_set_all_xy = rangeNormalize(test_set_all_xy, 0, 1)
elif NORMALIZE[0] == 'z':                                       #zscore normalizing
    train_set_all_xy = stats.zscore(train_set_all_xy)
    test_set_all_xy = stats.zscore(test_set_all_xy)
#

###creating training and validation set
train_data_xy = train_set_all_xy[:2170]             #creating training data 60%
train_data_label_xy = train_label_all_xy[:2170]     #creating training labels 60%

validation_set_xy = train_set_all_xy[-350:]         #creating validation data 10%
validation_label_xy = train_label_all_xy[-350:]     #creating validation labels 10%

###Exports processed data to dataPreProcess.bin
with open(PATH_EXP+'INPUT/PROCESSED/dataPreProcess.bin', 'wb') as file:
    for i in train_set_all_xy:
        file.write(bytes(i))                   #writes to bin
    for j in test_set_all_xy:
        file.write(bytes(j))                   #writes to bin

'''~~~~ TRAINING ~~~~'''
#your code to train the model. Export trained model parameters to load later as model.bin
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
EPOCHS = 100
#setting up checkpoint to save the best set of weights during the training
checkpoint_filepath = PATH_EXP+'MODEL/weights.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

#training model and saving history
#saving history is required to plot performance png
model_history = model.fit(train_data_xy, train_data_label_xy, batch_size=32, epochs=EPOCHS,\
                          validation_data=(validation_set_xy, validation_label_xy),\
                          callbacks=[model_checkpoint_callback], verbose=2)

#The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

#Exporting Model as h5 file
model.save(PATH_EXP+'MODEL/model.h5')

'''~~~~ TESTING ~~~~'''
#your code to test the model. Export predictions with sample-IDs as testOutput.bin
#setting dependencies for the model to be able to load the custom metrics
dependencies = {
    'AUC': AUC,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'f1': f1
}
#Importing Model
model = keras.models.load_model(PATH_EXP+'MODEL/model.h5', custom_objects=dependencies)

#predicted test values (1092) 1 nodes per prediction need to round and reshape for labels
test_pred_all_xy = []
for i in model.predict(test_set_all_xy).reshape(1092):  #reshape from (1092, 1) to (1092)
    test_pred_all_xy.append(round(i))                   #rounding to either 0 or 1

###Exports test output to testOutput.bin
with open(PATH_EXP+'OUTPUT/testOutput.bin', 'wb') as file:
    for i in test_pred_all_xy:
        file.write(bytes(i))                   #writes to bin
###

'''~~~~ EVALUATION ~~~~'''
#your code to evaluate performance of the model.
#Export performance metrics as csv and binary file
#performance.csv and performance.bin
#Calculate F1-score
f1score = f1_score(test_label_all_xy, test_pred_all_xy, average='weighted')
print('F1-score = {}'.format(f1score))
#Calculate AUC of ROC
aucRoc = roc_auc_score(test_label_all_xy, test_pred_all_xy)
print('AUC of ROC: {}'.format(aucRoc))
#Calculate Confusion Matrix
confMatrix = confusion_matrix(test_label_all_xy, test_pred_all_xy)
print('Confusion Matrix : \n', confMatrix)
#Calculate Sensitivity
sensitivity = confMatrix[0, 0]/(confMatrix[0, 0]+confMatrix[0, 1])
print('Sensitivity: {}'.format(sensitivity))
#Calculate Specificity
specificity = confMatrix[1, 1]/(confMatrix[1, 0]+confMatrix[1, 1])
print('Specificity: {}'.format(specificity))

###
#creating performance data dict and saving as csv and bin file
performance_dict = {'F1-score':f1score, 'AUC of ROC':aucRoc,\
                    'Sensitivity':sensitivity, 'Specificity':specificity}
save_csv_bin(PATH_EXP+'OUTPUT/performance.csv', PATH_EXP+'OUTPUT/performance.bin', performance_dict)
###

'''~~~~ VISUALIZE ~~~~'''
#your code to visualize performance metrics. Export charts.
performance = [f1score, aucRoc, sensitivity, specificity]   #creating performance list

#Using plot_train_history function to plot model performance
plot_train_history(model_history, performance)
#####

print('Done!')
