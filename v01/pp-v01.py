# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:47:28 2020

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Editing dates
def generalize(x):
    title = x['Date']
    s = ''
    t = title.split('-')
    s = s.join(t)
    return int(s)

dataset['Date2'] = dataset.apply(generalize, axis=1)
# Statistics on Afghanistan
dataset[dataset['Country_Region']=='Afghanistan']['ConfirmedCases'].plot.hist()


dataset[dataset['Country_Region']=='Afghanistan'].plot(kind='scatter', x='Date2', y='ConfirmedCases')

