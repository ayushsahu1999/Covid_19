# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:09:43 2020

@author: Dell
"""


# Importing the libraries
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()
countries = os.listdir(cwd)
countries.remove('pp-v01.py')
countries.remove('rnn-v01.py')
countries.remove('submission.csv')
countries.remove('test.csv')
countries.remove('train.csv')

for country in countries:
    fp = os.path.join(cwd, country)
    states = os.listdir(fp)
    for state in states:
        p1 = os.path.join(fp, state)
        log = os.path.join(p1, 'logs')
        if not os.path.exists(log):
            os.mkdir(log)
        else:
            continue
        # Importing the dataset
        dataset = pd.read_csv(os.path.join(p1, 'dataset.csv'))
        X = dataset.iloc[:, 1:5].values
        C = dataset.iloc[:, 5:6].values
        F = dataset.iloc[:, 6:7].values
        
        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        trained_cases_scaled = sc.fit_transform(C)
        trained_fatalities_scaled = sc.fit_transform(F)
        
        # Creating a data structure with 20 timestamps and 1 output
        CX_train = []
        CY_train = []
        #FX_train = []
        #FY_train = []
        for i in range(20, 67):
            CX_train.append(trained_cases_scaled[i-20:i, 0])
            CY_train.append(trained_cases_scaled[i, 0])
        #    FX_train.append(trained_fatalities_scaled[i-20:i, 0])
        #    FY_train.append(trained_fatalities_scaled[i-20:i, 0])
            
        CX_train, CY_train = np.array(CX_train), np.array(CY_train)
        #FX_train, FY_train = np.array(FX_train), np.array(FY_train)
        
        # Reshaping
        CX_train = np.reshape(CX_train, (CX_train.shape[0], CX_train.shape[1], 1))
        CX_val, CY_val = CX_train[30:47, :, :], CY_train[30:]
        CX_train, CY_train = CX_train[0:30, :, :], CY_train[0:30]
        #FX_train = np.reshape(FX_train, (FX_train.shape[0], FX_train.shape[1], 1))# remember to add one more layer

        # Initialize the model
        model = Sequential()
        
        # Adding the first LSTM layer and some Dropout Regularization
        model.add(LSTM(units=30, return_sequences=True, input_shape=(CX_train.shape[1], 1)))
        model.add(Dropout(0.2))
        
        # Adding the second LSTM layer and some Dropout Regularization
        model.add(LSTM(units=30, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Adding the third LSTM layer and some Dropout Regularization
        model.add(LSTM(units=30, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Adding the fourth LSTM layer and some Dropout Regularization
        model.add(LSTM(units=30))
        model.add(Dropout(0.2))
        
        # Adding the output layer
        model.add(Dense(units=1, activation='sigmoid'))
        
        #model.add(Dense(units=1))
        
        # Compiling the RNN
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fitting RNN to the training set
        model.fit(CX_train, CY_train, epochs=100, batch_size=32)
        
        model.save_weights(os.path.join(log, 'trained_weights.h5'))
        print ('Model saved at: '+str(p1))