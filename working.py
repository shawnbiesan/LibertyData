# -*- coding: utf-8 -*-
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from numpy import dtype
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor, Lasso, LinearRegression
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR, LinearSVR
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class CustomPipeline(object):
    @classmethod
    def get_pipeline(cls):
        pipe_clf = Pipeline([
        ('svd', TruncatedSVD(n_components=200)),
        ('mod', SGDRegressor())
        #('svr', RandomForestRegressor(n_estimators=10, n_jobs=-1))
        ])
        return pipe_clf
    
    @classmethod
    def get_pipeline_svr(cls):
        pipe_clf = Pipeline([
        ('svd', TruncatedSVD(n_components=200)),
        ('mod', LinearSVR(C=1, epsilon=.01))
        ])
        return pipe_clf    
    
    @classmethod
    def get_pipeline_tree(cls):
        pipe_clf = Pipeline([
        ('svd', TruncatedSVD(n_components=100)),
        ('mod', RandomForestRegressor(n_estimators=100, n_jobs=-1))
        ])
        return pipe_clf

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true
    
def validate_model_func(df, y, p, clf):
    df = df[0:int(df.shape[0] * p)]
    y = y[0:int(y.shape[0] * p)]
    gini_scorer = make_scorer(Gini)
    model = CustomPipeline.get_pipeline()
    print cross_validation.cross_val_score(model, df, y,
                                    scoring = gini_scorer, cv=4, n_jobs=-1)

def get_optimal_params(df, y, p, clf):
    df = df[0:int(df.shape[0] * p)]
    y = y[0:int(y.shape[0] * p)]
    gini_scorer = make_scorer(Gini)
    params = {#'mod__alpha': [1, .01, .001, .0001],
              'mod__loss': ['squared_epsilon_insensitive',
                            'epsilon_insensitive'],
              'mod__epsilon': [.1, 0, .01],
              #'mod__penalty': ['l2', 'l1', 'elasticnet']
              }
    model = grid_search.GridSearchCV(estimator=clf,
                                     param_grid=params,
                                     scoring=gini_scorer,
                                     cv=4,
                                     n_jobs=-1)
    model.fit(df, y)
    print model.best_score_
    print model.grid_scores_
    print model.best_estimator_

#Deprecated for above   
#def validate_model(df, y, p):
#    df = df[0:int(df.shape[0] * p)]
#    cv = cross_validation.KFold(df.shape[0], 10)
#    results = []
#    for traincv, testcv in cv:
#        train_ = df[traincv]
#        test_ = df[testcv]
#        y_train = y.values[traincv]
#        y_test = y.values[testcv]
#    
#        model = CustomPipeline.get_pipeline()
#        model.fit(train_, y_train)
#        
#        results.append(Gini(y_test, model.predict(test_)))
#        
#    print "final score: %s with std %s" %(str(np.mean(results)), str(np.std(results)),)
 

def test_model_holdout(df, y, p, clf):
    print clf
    train_ = df[0:int(df.shape[0] * p)]
    test_ = df[int(df.shape[0] * p):]
    print train_.shape, test_.shape
    
    y_train = y[0:int(y.shape[0] * p)]
    y_test = y[int(y.shape[0] * p):]
    print y_train.shape, y_test.shape
    
   # model = CustomPipeline.get_pipeline()
    #model.fit(train_, y_train)
    clf.fit(train_, y_train)
    
    print "model holdout value: " + str(Gini(y_test, clf.predict(test_)))
    print 
    return clf
    
    
if __name__ == '__main__':
    
    train = pd.read_csv('data/train.csv')
    train = train[0: int(train.shape[0] * 1)]
    
    test = pd.read_csv('data/test.csv')
    result = pd.read_csv('data/sample_submission.csv')
    y = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    
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
    #get_optimal_params(train, y, .75, CustomPipeline.get_pipeline_tree())

   # validate_model_func(train, y, .75)
    
    model1 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline())
    model2 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline_tree())
    
    stacked_predictions_train = pd.concat([pd.DataFrame(model1.predict(train),columns=['model1']),
                           pd.DataFrame(model2.predict(train),columns=['model2'])],
                            axis=1)
    stacked_model = test_model_holdout(stacked_predictions_train, y, .75, LinearRegression())
    
    stacked_predictions_test = pd.concat([pd.DataFrame(model1.predict(test),columns=['model1']),
                           pd.DataFrame(model2.predict(test),columns=['model2'])],
                            axis=1)
            
    
    result.Hazard = stacked_model.predict(stacked_predictions_test)
    result.to_csv('output.csv', index=False)
