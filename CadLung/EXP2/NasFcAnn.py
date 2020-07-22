'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: July 19 2020

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


class NasFcAnn(object):
    '''Attributes'''
    #Add class attributes simple to complex.
    #The following is just a partial list with default values. Add as needed.
    name = 'default'

    #Path Parameters    #'INPUT/Voi_Data/', 'EXP1/','EXP1/CHKPNT/'
    paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT')

    #Data Parameters
    normalize = 'none' #Options: {Default: 'none', 'ra', 'zs', 'pr'}

    #Model Parameters
    valSplit = 0.15
    epochs = 100
    batchSize = 32
    METRICS = ['accuracy']#, AUC(), Recall(), Precision(), FalsePositives(), TrueNegatives()]
    lossFn = 'mean_squared_error'
    optMthd = 'SGD'
    regRate = 0.001

    #Network Architecture Parameters
    #first layer, hidden layers, and output layer. #hidden notes > 1.
    maxNumNodes = (None, 6, 6, 6, 1)
    activationFn = (None, 'sigmoid', 'sigmoid', 'sigmoid', 'relu')

    #derived attributes
    regFn = rg.l2(regRate)
    checkpoint_filepath = paths[2]+'/checkpoint.hdf5'
    lenMaxNumHidenLayer = len(maxNumNodes) - 2
    modelChkPnt_cBk = cB.ModelCheckpoint(filepath=checkpoint_filepath,
                                         save_weights_only=True,
                                         monitor='val_accuracy',
                                         mode='max',
                                         save_best_only=True)
    clBacks = [modelChkPnt_cBk]
    #

    def __init__(self, **kwarg):
        '''Initialization'''
        self.name = kwarg['name']
        self.normalize = kwarg['normalize']
    #

    def exportParam(self, **kwarg):
        '''function to export parameters to csv file'''
        parameter_dict = {'Name': self.name, 'Normalization': self.normalize}
        with open(self.paths[1]+'/MODEL/{}Parameters.csv'.format(self.name), 'w') as file:
            for key in parameter_dict.keys():
                file.write("%s,%s\n"%(key, parameter_dict[key]))
        pass
    #

    def loadData(self, **kwarg):
        '''function to load data'''
        #VOI Data
        DataSet_xy = 'data_balanced_6Slices_1orMore_xy'
        #DataSet_xz = 'data_balanced_6Slices_1orMore_xz'
        #DataSet_yz = 'data_balanced_6Slices_1orMore_yz'

        with open(self.paths[0] + DataSet_xy + '.bin', 'rb') as f2:
            self.train_set_all_xy = np.load(f2)
            self.train_label_all_xy = np.load(f2)
            self.test_set_all_xy = np.load(f2)
            self.test_label_all_xy = np.load(f2)
        pass
    #

    def exportData(self, **kwarg):
        '''function to export data to bin file'''
        data_dict = {'Training Data': self.train_set_all_xy, 'Testing Data': self.test_set_all_xy}
        with open(self.paths[1]+'/INPUT/{}Data.bin'.format(self.name), 'wb') as file:
            for key in data_dict.keys():
                file.write(key.encode('ascii'))
                file.write(bytes(data_dict[key]))
        pass
    #

    def doPreProcess(self, **kwarg):
        '''function to do pre-processing on the data'''
        #Normalize your data
        def rangeNormalize(data, lower, upper): #lower, upper = range
            """function to range normalize data"""
            scaler = MinMaxScaler(feature_range=(lower, upper))
            normalized = scaler.fit_transform(data)
            return normalized
        #

        def positiveNormalize(data):
            """function to move data to the positive region"""
            for i in range(len(data)):
                dataMin = min(data[i])
                if dataMin < 0.0001:
                    scal = 0.0001-dataMin
                    data[i] = data[i] + scal    #shifting elements to make minimum 0.0001
            return data
        #

        if self.normalize == 'ra':
            self.train_set_all_xy = rangeNormalize(self.train_set_all_xy, 0, 1)
            self.test_set_all_xy = rangeNormalize(self.test_set_all_xy, 0, 1)
        elif self.normalize == 'zs':
            self.train_set_all_xy = stats.zscore(self.train_set_all_xy)
            self.test_set_all_xy = stats.zscore(self.test_set_all_xy)
        elif self.normalize == 'pr':
            self.train_set_all_xy = positiveNormalize(self.train_set_all_xy)
            self.test_set_all_xy = positiveNormalize(self.test_set_all_xy)
        #

        #Dim1: Batch, (Dim2,Dim3): Flattened images
        self.train_set_all_xy = np.reshape(self.train_set_all_xy,
                                           (self.train_set_all_xy.shape[0],
                                            1, self.train_set_all_xy.shape[1]))

        #Dim1: Batch, (Dim2,Dim3): Flattened images
        self.test_set_all_xy = np.reshape(self.test_set_all_xy,
                                          (self.test_set_all_xy.shape[0],
                                           1, self.test_set_all_xy.shape[1]))
    #

    def exportPreProcData(self, **kwarg):
        '''function to export pre processed data to bin file'''
        preProcessedData_dict = {'Processed Training Data': self.train_set_all_xy,
                                 'Processed Testing Data': self.test_set_all_xy}

        with open(self.paths[1]+'/INPUT/PROCESSED/{}PreProcessedData.bin'.format(self.name), 'wb') as file:
            for key in preProcessedData_dict.keys():
                file.write(key.encode('ascii'))
                file.write(bytes(preProcessedData_dict[key]))
        pass
    #

    def setUpModelTrain(self, **kwarg):
        '''function to find the best model structure'''
        bestAcc = 0 #best accuracy score
        #[0 0 0 0] 1st one is for input. number of nodes added to the current layer
        numNodeLastHidden = np.zeros(self.lenMaxNumHidenLayer + 1)

        #Searching the best network architecture
        for hL in range(1, self.lenMaxNumHidenLayer+1):  #Hidden Layer Loop (1 to 4)
            for j in range(2, self.maxNumNodes[hL]):   #Node loop   (2 to 6), 3 times
                numNodeLastHidden[hL] += 1  #A new node added to the current layer [0 0 0]
                #Re-create the temp model with a new node at the layer
                modelTmp = keras.Sequential()   #initialize temporary model
                modelTmp.add(Flatten())         #Input layer

                for iL in range(1, hL+1):             #Adds number of hidden layers
                    modelTmp.add(Dense(int(numNodeLastHidden[iL]), activation=self.activationFn[hL],
                                       kernel_initializer='random_normal', kernel_regularizer=self.regFn))

                #output layer
                modelTmp.add(Dense(1, activation=self.activationFn[-1], kernel_initializer='random_normal',
                                   kernel_regularizer=self.regFn))


                modelTmp.compile(loss=self.lossFn, optimizer=self.optMthd, metrics=self.METRICS)
                modelFitTmp = modelTmp.fit(self.train_set_all_xy, self.train_label_all_xy,
                                           batch_size=self.batchSize,
                                           epochs=self.epochs, verbose=0, callbacks=self.clBacks,
                                           validation_split=self.valSplit)

                #After pulling out the best weights and the corresponding model "modelFitTmp",
                #compare against the last "bestAcc"

                if max(modelFitTmp.history['val_accuracy']) > bestAcc:  #must be just '>'
                    print(max(modelFitTmp.history['val_accuracy']))
                    #update the best model and continue adding a node to this layer
                    bestAcc = max(modelFitTmp.history['val_accuracy'])
                    self.bestModel = modelTmp
                    del modelTmp    #WHY ?
                else:   #adding a new node did not improve the performance.
                    #Stop adding a new node to this layer
                    break
                #
            #for j
        #for hL
        pass
    #

    def exportModel(self, **kwarg):
        '''function to save model to hdf5 file'''
        self.bestModel.load_weights(self.checkpoint_filepath)   #loading best weights
        self.bestModel.save(self.paths[1]+'/MODEL/{}Model.hdf5'.format(self.name))  #saving best model
        pass
    #

    def testModel(self, **kwarg):
        #making test prediction
        self.test_pred = []
        #reshape from (1092, 1) to (1092)
        for i in self.bestModel.predict(self.test_set_all_xy).reshape(self.test_label_all_xy.shape[0]):
            self.test_pred.append(round(i))
        testAcc = accuracy_score(self.test_label_all_xy, self.test_pred)  #test accuracy

        print(self.bestModel.summary())  #Printing new model structure
        print('Test Accuracy: {}'.format(testAcc))
    #

    def exportPredict(self, **kwarg):
        '''function to export model predictions to bin file'''
        with open(self.paths[1]+'/OUTPUT/{}TestPredictions.bin'.format(self.name), 'wb') as file:
            for i in self.test_pred:
                file.write(bytes(i))
        pass
    #

    def evaluate(self, **kwarg):
        '''function to evaluate performance of model'''
        #Evaluate performance of the model
        #Calculate F1-score
        self.f1score = f1_score(self.test_label_all_xy, self.test_pred, average='weighted')
        print('F1-score = {}'.format(self.f1score))
        #Calculate AUC of ROC
        self.aucRoc = roc_auc_score(self.test_label_all_xy, self.test_pred)
        print('AUC of ROC: {}'.format(self.aucRoc))
        #Calculate Confusion Matrix
        confMatrix = confusion_matrix(self.test_label_all_xy, self.test_pred)
        print('Confusion Matrix : \n', confMatrix)
        #Calculate Sensitivity
        self.sensitivity = confMatrix[0, 0]/(confMatrix[0, 0]+confMatrix[0, 1])
        print('Sensitivity: {}'.format(self.sensitivity))
        #Calculate Specificity
        self.specificity = confMatrix[1, 1]/(confMatrix[1, 0]+confMatrix[1, 1])
        print('Specificity: {}'.format(self.specificity))

        pass
    #

    def exportTestPerf(self, **kwarg):
        '''function to export test performance to csv file'''
        testPerformance_dict = {'F1 Score': self.f1score, 'AUC': self.aucRoc,
                                'Sensitivity': self.sensitivity, 'Specificity': self.specificity}

        with open(self.paths[1]+'/OUTPUT/{}TestPerformance.csv'.format(self.name), 'w') as file:
            for key in testPerformance_dict.keys():
                file.write("%s,%s\n"%(key, testPerformance_dict[key]))
        pass
    #

    def visualPerf(self, **kwarg):
        '''function to visualize model performance'''
        performance = [self.f1score, self.aucRoc, self.sensitivity, self.specificity]
        x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']    #creating performance labels
        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, performance, color=('green', 'red', 'blue', 'yellow'))
        plt.ylim([0, 1])     #setting performance score range
        plt.xticks(x_pos, x)
        plt.title('Model Performance Metrics')

        plt.show()
        pass
    #

    def exportChart(self, **kwarg):
        '''function to save chart as a png file'''
        performance = [self.f1score, self.aucRoc, self.sensitivity, self.specificity]
        x = ['F1-score', 'AUC of ROC', 'Sensitivity', 'Specificity']    #creating performance labels
        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, performance, color=('green', 'red', 'blue', 'yellow'))
        plt.ylim([0, 1])     #setting performance score range
        plt.xticks(x_pos, x)
        plt.title('Model Performance Metrics')

        plt.savefig(self.paths[1]+'/OUTPUT/{}ModelPerformance.png'.format(self.name))
        pass
    #
