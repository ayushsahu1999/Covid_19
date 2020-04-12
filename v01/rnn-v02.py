# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:03:20 2020

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dataset = pd.read_csv('Afghanistan/nan/dataset.csv')
X = dataset.iloc[:, 1:5].values
C = dataset.iloc[:, 5:6].values
F = dataset.iloc[:, 6:7].values

#for i in range(14472, 14539):
#    X[i][2] = X[i][2][:-1]
    
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
#FX_train = np.reshape(FX_train, (FX_train.shape[0], FX_train.shape[1], 1))# remember to add one more layer

# Buid the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

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

#model.add(Flatten())
# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

#model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
# Fitting RNN to the training set
model.fit(CX_train, CY_train, epochs=50, batch_size=32)

model.save_weights('Afghanistan/nan/logs/trained_weights.h5')