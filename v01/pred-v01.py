# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:02:32 2020

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Importing the dataset
dataset = pd.read_csv('Afghanistan/nan/test.csv')
X = dataset.iloc[:, 1:5].values
train = pd.read_csv('Afghanistan/nan/dataset.csv')

ds = train.iloc[47:, 5:6].values

    
# Initialize the model
model = Sequential()

# Adding the first LSTM layer and some Dropout Regularization
model.add(LSTM(units=30, return_sequences=True, input_shape=(20, 1)))# as CX_train.shape[1] is timesteps which is 20.
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

model_path = 'Afghanistan/nan/logs/trained_weights.h5'
model.load_weights(model_path, by_name=True)


test = np.copy(ds)
preds = []
scaled = open("Afghanistan/nan/scaled.pickle", "rb")
sc = pickle.load(scaled)
test = sc.transform(test)

for i in range(20, 63):
    X_test = test[i-20:i, 0]
    
    CX_test = np.reshape(X_test, (1, 20, 1))
    pred = model.predict(CX_test)
    test = np.append(test, pred[0][0])
    test = test.reshape(-1, 1)
    
    pred = sc.inverse_transform(pred)
    
    preds.append(pred[0][0])
    
for p in preds:
    print (p)
