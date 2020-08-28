'''
SELU - CAR System -- LBRN 2020 Virtual Summer Program
Mentor: Dr. Omer Muhammet Soysal
Last Modified on: August 27 2020

Authors: William Brown
Date:
Distribution: Participants of SELU - CAR System -- LBRN 2020 Virtual Summer Program

CODING NOTES
- Use relative path
- Be sure your code is optimized as much as possible.
- Read the coding standarts and best practice notes.
~~~~ HELP  ~~~~
- Class to create models
'''
import os
import keras

from keras import callbacks as cB
from keras import regularizers as rg
from keras.layers import Dense, Flatten
from keras.metrics import AUC
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np


class NasFcAnn(object):
    '''Attributes'''
    #Add class attributes simple to complex.
    #The following is just a partial list with default values. Add as needed.
    __name = 'default'
    __type = 'slice'

    #Path Parameters    #'INPUT/Voi_Data/', 'EXP1/','EXP1/CHKPNT/'
    paths = ('CadLung/INPUT/Voi_Data/', 'CadLung/EXP2/', 'CadLung/EXP2/CHKPNT/')
    __dataPath = 'none'

    #Data Parameters
    __normalize = 'none' #Options: {Default: 'none', 'ra', 'zs'}
    __positiveRegion = 'y' #Options: {Default: 'n', 'y'}
    positiveRegionMin = 0.0001 #{Default: 0.0001}

    #Model Parameters
    valSplit = 0.15
    epochs = 100
    batchSize = 32
    METRICS = ['accuracy', AUC()]
    lossFn = 'binary_crossentropy'
    initializer = 'random_normal'
    optMthd = 'adam'
    __regRate = 0.001

    #Network Architecture Parameters
    #first layer, hidden layers, and output layer. #hidden notes > 1.
    maxNumNodes = (None, 6, 6, 6, 1)
    activationFn = (None, 'relu', 'relu', 'relu', 'sigmoid')

    #derived attributes
    regFn = rg.l2(__regRate)
    checkpoint_filepath = paths[2]+'checkpoint.hdf5'
    lenMaxNumHidenLayer = len(maxNumNodes) - 2
    modelChkPnt_cBk = cB.ModelCheckpoint(filepath=checkpoint_filepath,
                                         save_weights_only=True,
                                         monitor='val_accuracy',
                                         mode='max',
                                         save_best_only=True)
    clBacks = [modelChkPnt_cBk]

    def __init__(self, **kwarg):
        '''Initialization'''
        self.__dataPath = kwarg['dataPath']
        self.__name = kwarg['name']
        self.__type = kwarg['type']
        self.__normalize = kwarg['normalize']
        self.__regRate = kwarg['regRate']
        self.__positiveRegion= kwarg['positiveRegion']
    #

    def exportParam(self, **kwarg):
        '''function to export parameters to csv file'''
        parameter_dict = {'Name': self.__name, 'Normalization': self.__normalize}
        with open(self.paths[1]+'/MODEL/{}Parameters.tf'.format(self.__name), 'w') as file:
            for key in parameter_dict.keys():
                file.write("%s,%s\n"%(key, parameter_dict[key]))
    #

    def loadData(self, **kwarg):
        '''function to load data'''
        if self.__type == 'slice':
            dataset_filepath = self.paths[0]+self.__dataPath
            with open(dataset_filepath, 'rb') as f2:
                self.train_set_all = np.load(f2)
                self.train_label_all = np.load(f2)
                self.test_set_all = np.load(f2)
                self.test_label_all = np.load(f2)
        else:
            file = np.load(self.__dataPath)
            self.train_set_all = file['arr_0']
            self.train_label_all = file['arr_1']
            self.test_set_all = file['arr_2']
            self.test_label_all = file['arr_3']
    #

    def exportData(self, **kwarg):
        '''function to export data to bin file'''
        data_list = [self.train_set_all, self.test_set_all]
        with open(self.paths[1]+'/INPUT/{}Data.tf'.format(self.__name), 'w') as file:
            for i in data_list[0]:
                file.write(str(i)+'\n')
            for j in data_list[1]:
                file.write(str(j)+'\n')
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
                if dataMin < self.positiveRegionMin:
                    scal = self.positiveRegionMin-dataMin
                    data[i] = data[i] + scal    #shifting elements to make minimum 0.0001
            return data
        #

        if self.__positiveRegion == 'y':
            self.train_set_all = positiveNormalize(self.train_set_all)
            self.test_set_all = positiveNormalize(self.test_set_all)

        if self.__normalize == 'ra':
            self.train_set_all = rangeNormalize(self.train_set_all, 0, 1)
            self.test_set_all = rangeNormalize(self.test_set_all, 0, 1)
        elif self.__normalize == 'zs':
            self.train_set_all = stats.zscore(self.train_set_all)
            self.test_set_all = stats.zscore(self.test_set_all)
        #

        #Dim1: Batch, (Dim2,Dim3): Flattened images
        self.train_set_all = np.reshape(self.train_set_all,
                                           (self.train_set_all.shape[0],
                                            1, self.train_set_all.shape[1]))

        #Dim1: Batch, (Dim2,Dim3): Flattened images
        self.test_set_all = np.reshape(self.test_set_all,
                                          (self.test_set_all.shape[0],
                                           1, self.test_set_all.shape[1]))
    #

    def exportPreProcData(self, **kwarg):
        '''function to export pre processed data to bin file'''
        preProcessedData_list = [self.train_set_all, self.test_set_all]

        with open(self.paths[1]+'INPUT/PROCESSED/{}PreProcessedData.tf'.format(self.__name), 'w') as file:
            for i in preProcessedData_list[0]:
                file.write(str(i)+'\n')
            for j in preProcessedData_list[1]:
                file.write(str(i)+'\n')
    #

    def setUpModelTrain(self, **kwarg):
        '''function to find the best model structure'''
        bestAcc = 0 #best accuracy score
        self.bestModel = keras.Sequential([
        Flatten(),
        Dense(6, activation=self.activationFn[1], kernel_initializer=self.initializer,
              kernel_regularizer=self.regFn), #1st hidden layer
        Dense(1, activation=self.activationFn[-1], kernel_initializer=self.initializer,
              kernel_regularizer=self.regFn) #output layer
        ])

        self.bestModel.compile(loss=self.lossFn, optimizer=self.optMthd, metrics=['accuracy', AUC()])
        modelFit = self.bestModel.fit(self.train_set_all, self.train_label_all,
                                 batch_size=self.batchSize, epochs=self.epochs,
                                 verbose=0, callbacks=self.clBacks,
                                 validation_split=self.valSplit)

        self.bestModel.load_weights(self.checkpoint_filepath, by_name=True, skip_mismatch=True)
        bestLoss, bestAcc, bestAUC = self.bestModel.evaluate(self.test_set_all, self.test_label_all, verbose=0)

        #[0 0 0 0] 1st one is for input. number of nodes added to the current layer
        numNodeLastHidden = np.zeros(self.lenMaxNumHidenLayer + 1)

        #Searching the best network architecture
        for hL in range(1, self.lenMaxNumHidenLayer+1):  #Hidden Layer Loop (1 to 4)
            for j in range(1, self.maxNumNodes[hL]+1):   #Node loop   (1 to 6), 3 times
                numNodeLastHidden[hL] += 1  #A new node added to the current layer
                #Re-create the temp model with a new node at the layer
                modelTmp = keras.Sequential()   #initialize temporary model
                modelTmp.add(Flatten())         #Input layer

                for iL in range(1, hL+1):             #Adds number of hidden layers
                    modelTmp.add(Dense(int(numNodeLastHidden[iL]), activation=self.activationFn[hL],
                                       kernel_initializer=self.initializer, kernel_regularizer=self.regFn))

                #output layer
                modelTmp.add(Dense(1, activation=self.activationFn[-1], kernel_initializer=self.initializer,
                                   kernel_regularizer=self.regFn))


                modelTmp.compile(loss=self.lossFn, optimizer=self.optMthd, metrics=['accuracy', AUC()])
                modelFitTmp = modelTmp.fit(self.train_set_all, self.train_label_all,
                                           batch_size=self.batchSize,
                                           epochs=self.epochs, verbose=0, callbacks=self.clBacks,
                                           validation_split=self.valSplit)

                #After pulling out the best weights and the corresponding model "modelFitTmp",
                #After pulling out the best weights and the corresponding model "modelFitTmp",
                modelTmp.load_weights(self.checkpoint_filepath, by_name=True, skip_mismatch=True)    #loading test weights
                #modelTmp test evaluation
                tmpLoss, tmpAcc, tmpAUC = modelTmp.evaluate(self.test_set_all, self.test_label_all, verbose=0)
                #compare against the last "bestAcc"

                if tmpAcc > bestAcc:
                    #update the best model and continue adding a node to this layer
                    print(tmpAcc)
                    bestAcc = tmpAcc
                    self.bestModel = modelTmp
                    del modelTmp    #WHY ?
                else:   #adding a new node did not improve the performance.
                    if numNodeLastHidden[hL] != 1:
                        numNodeLastHidden[hL] -= 1  #going back to best number of nodes
                    #Stop adding a new node to this layer
                    break
                #
            #for j
        #for hL
        print(self.bestModel.summary())
    #

    def exportModel(self, **kwarg):
        '''function to save model to hdf5 file'''
        self.bestModel.load_weights(self.checkpoint_filepath, by_name=True, skip_mismatch=True)   #loading best weights
        self.bestModel.save(self.paths[1]+'/MODEL/{}Model.hdf5'.format(self.__name))  #saving best model
    #

    def loadModel(self, **kwarg):
        self.bestModel = keras.models.load_model(self.paths[1]+'/MODEL/{}Model.hdf5'.format(self.__name),
                                                custom_objects={'AUC': AUC()}, compile=False)
        self.bestModel.compile(loss=self.lossFn, optimizer=self.optMthd, metrics=['accuracy', AUC()])
        #

    def testModel(self, **kwarg):
        #making test prediction
        self.test_pred = []
        #reshape from (1092, 1) to (1092)
        for i in self.bestModel.predict(self.test_set_all).reshape(self.test_label_all.shape[0]):
            if self.__type == 'volume':
                self.test_pred.append(i.round())
            else:
                self.test_pred.append(i)
    #

    def exportPredict(self, **kwarg):
        '''function to export model predictions to tf file'''
        with open(self.paths[1]+'OUTPUT/{}TestPredictions.tf'.format(self.__name), 'w') as file:
            for i in self.test_pred:
                file.write(str(i)+'\n')
    #

    def evaluate(self, **kwarg):
        '''function to evaluate performance of model'''
        #Evaluate performance of the model
        self.testLoss, self.testAcc, self.testAUC = self.bestModel.evaluate(self.test_set_all, self.test_label_all, verbose=0)
        print('Accuracy: {}'.format(self.testAcc))
        print('AUC of ROC: {}'.format(self.testAUC))
    #

    def exportTestPerf(self, **kwarg):
        '''function to export test performance to csv file'''
        testPerformance_dict = {'Accuracy': self.testAcc, 'AUC': self.testAUC}

        with open(self.paths[1]+'/OUTPUT/{}TestPerformance.csv'.format(self.__name), 'w') as file:
            for key in testPerformance_dict.keys():
                file.write("%s,%s\n"%(key, testPerformance_dict[key]))
    #

    def visualPerf(self, **kwarg):
        '''function to visualize model performance'''
        performance = [self.testAcc, self.testAUC]
        x = ['Accuracy', 'AUC of ROC']    #creating performance labels
        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, performance, color=('blue', 'red'))
        plt.ylim([0, 1])     #setting performance score range
        plt.xticks(x_pos, x)
        plt.title('Model Performance Metrics')

        plt.show()
    #

    def exportChart(self, **kwarg):
        '''function to save chart as a png file'''
        performance = [self.testAcc, self.testAUC]
        x = ['Accuracy', 'AUC of ROC']    #creating performance labels
        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, performance, color=('blue', 'red'))
        plt.ylim([0, 1])     #setting performance score range
        plt.xticks(x_pos, x)
        plt.title('Model Performance Metrics')

        plt.savefig(self.paths[1]+'/OUTPUT/{}ModelPerformance.png'.format(self.__name))
    #

    def exportTrainPred(self, **kwarg):
        '''function to export training predictions to the next Model'''
        #making train predictions
        train_pred = []
        for i in self.bestModel.predict(self.train_set_all).reshape(self.train_label_all.shape[0]):
            train_pred.append(i)
        with open(self.paths[1]+'OUTPUT/{}TrainPredictions.tf'.format(self.__name), 'w') as file:
            for i in train_pred:
                file.write(str(i)+'\n')
    #
