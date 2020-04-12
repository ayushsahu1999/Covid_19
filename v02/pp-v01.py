# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import pickle
# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:4].values
C = dataset.iloc[:, 4].values
F = dataset.iloc[:, 5].values

i = 0
j = 0
countries = []
states = []
s = []
c_name = X[0][2]

cwd = os.getcwd()
while (i < 67 and j < 19698):
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
            p1 = os.path.join(fp, X[j][1])
            s.append(X[j][1])
        if not os.path.exists(p1):
            os.mkdir(p1)
    i = i + 1
    j = j + 1
    i = i % 67
countries.append(c_name)
states.append(s)
    



#fp = ''
#p1 = ''
#p2 = ''

#countries = os.listdir(cwd)
#countries.remove('covid19-global-forecasting-week-2.zip')
#countries.remove('pp-v01.py')
#countries.remove('rnn-v02.py')
#countries.remove('rnn-v03.py')
#countries.remove('pp-v02.py')
#countries.remove('submission.csv')
#countries.remove('test.csv')
#countries.remove('train.csv')
i = 0

for c, country in enumerate(countries):
    fp = os.path.join(cwd, country)
    #states = os.listdir(fp)
    for state in states[c]:
        p1 = os.path.join(fp, state)
        
        if os.path.exists(os.path.join(p1, 'dataset.csv')):
            continue
        
        
        if (str(X[i][1]) == state and X[i][2] == country):
            # Save dataset here
            
            #tX = X[i:i+67]
            #tC = C[i:i+67]
            #tF = F[i:i+67]
            #tD = np.column_stack((tX, tC, tF))
            p2 = os.path.join(p1, 'dataset.csv')
            #np.savetxt(p2, tD, delimiter=',')
            td = dataset[i:i+67]
            td.to_csv(p2)
            i = i + 67
        """
        else:
            for q in range(0, 19698):
                if (X[q][2] == country and str(X[q][1]) == state):
                    # Save dataset here
                    
                    #tX = X[i:i+67]
                    #tC = C[i:i+67]
                    #tF = F[i:i+67]
                    #tD = np.column_stack((tX, tC, tF))
                    p2 = os.path.join(p1, 'dataset.csv')
                    #np.savetxt(p2, tD, delimiter=',')
                    td = dataset[q:q+67]
                    td.to_csv(p2)
                    break
        """
            
# To save the scaled values
i = 0

for c, country in enumerate(countries):
    fp = os.path.join(cwd, country)
    #states = os.listdir(fp)
    for state in states[c]:
        p1 = os.path.join(fp, state)
        z1 = os.path.join(p1, 'dataset.csv')
        pick = os.path.join(p1, 'scaled.pickle')
        
        
        
        dataset = pd.read_csv(z1)
        
        C = dataset.iloc[:, 5:6].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        trained_cases_scaled = sc.fit_transform(C)
        #trained_fatalities_scaled = sc.fit_transform(F)
        
        scaled = open(pick, "wb")
        pickle.dump(sc, scaled)
        scaled.close()
        