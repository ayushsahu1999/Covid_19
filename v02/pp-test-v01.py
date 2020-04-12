# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:38:28 2020

@author: Dell
"""

# This python file is for testing dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math


# Importing the dataset
dataset = pd.read_csv('test.csv')
X = dataset.iloc[:, 0:4].values

i = 0
j = 0
countries = []
states = []
s = []
c_name = X[0][2]

cwd = os.getcwd()

while (i < 43 and j < 12642):
    if i == 0:
        if (X[j][2] == 'Taiwan*'):
            X[j][2] = X[j][2][:-1]
        fp = os.path.join(cwd, X[j][2])
        
        
        
        if (X[j][2] != c_name):
            countries.append(c_name)
            states.append(s)
            s = []
            c_name = X[j][2]
    if not os.path.exists(fp):
        os.mkdir(fp)
    try:
        if math.isnan(X[j][1]):
            if i == 0:
                p1 = os.path.join(fp, 'nan')
                s.append('nan')
            if not os.path.exists(p1):
                os.mkdir(p1)
    except:
        if i == 0:
            p1 = os.path.join(fp, str(X[j][1]))
            s.append(str(X[j][1]))
        if not os.path.exists(p1):
            os.mkdir(p1)
    i = i + 1
    j = j + 1
    i = i % 43
countries.append(c_name)
states.append(s)


i = 0

for c, country in enumerate(countries):
    fp = os.path.join(cwd, country)
    #states = os.listdir(fp)
    for state in states[c]:
        p1 = os.path.join(fp, state)
        
        if os.path.exists(os.path.join(p1, 'test.csv')):
            continue
        
        
        if (str(X[i][1]) == state and X[i][2] == country):
            
            p2 = os.path.join(p1, 'test.csv')
            #np.savetxt(p2, tD, delimiter=',')
            td = dataset[i:i+43]
            td.to_csv(p2)
            i = i + 43