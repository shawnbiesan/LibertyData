# -*- coding: utf-8 -*-
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from numpy import dtype
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR

# normalized gini by 0x0FFF
def gini(solution, submission):
    df = zip(solution, submission)
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini
    

train = pd.read_csv('data/train.csv')
train = train[0: int(train.shape[0] * 1)]
test = pd.read_csv('data/test.csv')
result = pd.read_csv('data/sample_submission.csv')
y = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

for col in train.columns:
    #if train[col].dtype == dtype('O'):
        #print "transforming col %s" %(col,)
        #train = pd.concat([train, pd.get_dummies(train[col], prefix=col)])
        #train.drop(col, axis=1, inplace=True)
    if train[col].dtype == dtype('O'):
        print "transforming col %s" %(col,)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[col]) + list(test[col]))
        train[col] = lbl.transform(train[col])
        test[col] = lbl.transform(test[col])
model = SGDRegressor()
model.fit(train, y)

result.Hazard = model.predict(test)
result.to_csv('output.csv', index=False)
