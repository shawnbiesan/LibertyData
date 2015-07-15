# -*- coding: utf-8 -*-
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from numpy import dtype
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion

class CustomPipeline(object):
    @classmethod
    def get_pipeline(cls):
        pipe_clf = Pipeline([
        ('svd', TruncatedSVD(n_components=500)),
        ('sgd', SGDRegressor())
        ])
        return pipe_clf
    

def gini(list_of_values):
  sorted_list = sorted(list(list_of_values))
  height, area = 0, 0
  for value in sorted_list:
    height += value
    area += height - value / 2.
  fair_area = height * len(list_of_values) / 2
  return (fair_area - area) / fair_area
  
def normalized_gini(y_pred, y):
    normalized_gini = gini(y_pred)/gini(y)
    return normalized_gini
    
def validate_model(df, y, p):
    df = df[0:int(df.shape[0] * p)]
    cv = cross_validation.KFold(df.shape[0], 10)
    results = []
    for traincv, testcv in cv:
        train_ = df[traincv]
        test_ = df[testcv]
        y_train = y.values[traincv]
        y_test = y.values[testcv]
    
        model = CustomPipeline.get_pipeline()
        model.fit(train_, y_train)
        
        results.append(normalized_gini(model.predict(test_), y_test))
        
    print "final score: %s with std %s" %(str(np.mean(results)), str(np.std(results)),)
 

def test_model_holdout(df, y, p):
    train_ = df[0:int(df.shape[0] * p)]
    test_ = df[int(df.shape[0] * p):]
    
    y_train = y[0:int(y.shape[0] * p)]
    y_test = y[int(y.shape[0] * p):]
    
    model = CustomPipeline.get_pipeline()
    model.fit(train_, y_train)
    
    print normalized_gini(model.predict(test_), y_test)
    
    

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
ohc = preprocessing.OneHotEncoder()
train = ohc.fit_transform(train)
test = ohc.transform(test)

########
validate_model(train, y, .75)

test_model_holdout(train, y, .75)


########

#svd = TruncatedSVD(n_components=500, random_state=42)
#train = svd.fit_transform(train)
#test = svd.transform(test)

#model = SGDRegressor()
#model.fit(train, y)

#result.Hazard = model.predict(test)
#result.to_csv('output.csv', index=False)
