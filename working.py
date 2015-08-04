# -*- coding: utf-8 -*-
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from numpy import dtype
from sklearn.linear_model import SGDRegressor, Lasso, LinearRegression
from sklearn import decomposition, pipeline, metrics, grid_search, preprocessing, cross_validation
from sklearn.svm import SVR, LinearSVR
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.manifold import TSNE
import xgboost as xgb
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from lasagne import layers




one_hot_columns = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8',
                   'T2_V11', 'T2_V12', 'T2_V13', 'T1_V9', 'T1_V11', 'T1_V12',
                   'T1_V15', 'T1_V16', 'T1_V17']

class CustomPipeline(object):
    @classmethod
    def get_pipeline(cls, with_svd=False):
        pipe_clf = Pipeline([
        #('poly', preprocessing.PolynomialFeatures()),
        #('svd', TruncatedSVD(n_components=100)),
        ('mod', SGDRegressor())
        ])
        return pipe_clf
    
    @classmethod
    def get_pipeline_svr(cls, with_svd=False):
        pipe_clf = Pipeline([
        #('poly', preprocessing.PolynomialFeatures()),
        #('svd', TruncatedSVD(n_components=100)),
        ('mod', LinearSVR(C=1, epsilon=.01))
        ])
        return pipe_clf    
    
    @classmethod
    def get_pipeline_rf(cls, with_svd=False):
        pipe_clf = Pipeline([
        #('poly', preprocessing.PolynomialFeatures()),
        #('svd', TruncatedSVD(n_components=100)),
        ('mod', RandomForestRegressor(n_estimators=50, n_jobs=-1))
        ])
        return pipe_clf

    @classmethod
    def get_pipeline_tree(cls, with_svd=False):
        pipe_clf = Pipeline([
        #('poly', preprocessing.PolynomialFeatures()),
        #('svd', TruncatedSVD(n_components=50)),
        ('mod', xgb.XGBRegressor(n_estimators=300, min_child_weight=8,
                                 subsample=.5, max_depth=3, learning_rate=.111)),
        ])
        return pipe_clf

    @classmethod
    def get_pipeline_neural(cls, with_svd=False):
        layers_s = [('input', layers.InputLayer),
           ('dense0', layers.DenseLayer),
           ('output', layers.DenseLayer)

        ]

        pipe_clf = Pipeline([
        ('mod',NeuralNet(layers=layers_s, input_shape=(None, train.shape[1]),
                         dense0_num_units=43, output_num_units=1, output_nonlinearity=None,
                         regression=True, update=nesterov_momentum, update_learning_rate=0.001,
                         update_momentum=0.9, eval_size=0.2, verbose=0, max_epochs=100)),
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
    params = {
                'mod__max_depth': [6, 7,],
                'mod__min_child_weight': [1, 3],
                'mod__subsample': [.5, .8, 1],
                #'mod__eta': [.03, .01],
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
    print train_.shape, y_train.shape
    print train
    clf.fit(train_, y_train)

    pred = clf.predict(test_)
    pred = np.reshape(pred, (pred.shape[0]))
    print y_test.shape, pred.shape
    print "model holdout value: " + str(Gini(y_test, pred))
    print 
    return clf

    
if __name__ == '__main__':
    
    train = pd.read_csv('data/train.csv')
    train = train[0: int(train.shape[0] * 1)]
    
    test = pd.read_csv('data/test.csv')
    result = pd.read_csv('data/sample_submission.csv')
    y = train.Hazard
    y = y.astype(np.float32)
    train.drop('Hazard', axis=1, inplace=True)
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    
    for col in train.columns:
        if train[col].dtype == dtype('O'):
            print "transforming col %s" %(col,)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[col]) + list(test[col]))
            train[col] = lbl.transform(train[col])
            test[col] = lbl.transform(test[col])
    ohc = preprocessing.OneHotEncoder(sparse=False)
    #train_hot = ohc.fit_transform(train)
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    train = train.astype(np.float32)
    test = scaler.fit_transform(test)
    test = test.astype(np.float32)
    

    #train = pd.concat([train, pd.DataFrame(tmp_train)], axis=1)
    #train.drop(one_hot_columns, axis=1, inplace=True)

    #test_hot = ohc.transform(test)
    #test = pd.concat([test, pd.DataFrame(tmp_test)], axis=1)
    #test.drop(one_hot_columns, axis=1, inplace=True)

    
    ########
    #get_optimal_params(train, y, .75, CustomPipeline.get_pipeline_tree())

   # validate_model_func(train, y, .75)

    ## todo, allow model 1 to use diff data, and also in general look at which ones should actually be ordinal

    model1 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline_tree())
    #model2 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline_tree())
    model3 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline_neural())
    model4 = test_model_holdout(train, y, .75, CustomPipeline.get_pipeline_rf())

    model_list = [model1, model3, model4]

    #stacked_predictions_train = pd.concat([pd.DataFrame(model.predict(train)) for model in model_list], axis=1)
    stacked_predictions_train = pd.concat([pd.DataFrame(model1.predict(train)),
                                 #pd.DataFrame(model2.predict(train)),
                                 pd.DataFrame(model3.predict(train)),
                                 pd.DataFrame(model4.predict(train)),
                                ], axis=1)
    stacked_predictions_train.columns = ['model' + str(i+1) for i in range(len(model_list))]

    stacked_model = test_model_holdout(stacked_predictions_train, y, .75, LinearRegression())
    
    #stacked_predictions_test = pd.concat([pd.DataFrame(model.predict(test)) for model in model_list], axis=1)
    stacked_predictions_test = pd.concat([pd.DataFrame(model1.predict(test)),
                                 #pd.DataFrame(model2.predict(test)),
                                 pd.DataFrame(model3.predict(test)),
                                 pd.DataFrame(model4.predict(test)),
                                ], axis=1)
    stacked_predictions_test.columns = ['model' + str(i+1) for i in range(len(model_list))]
            

    result.Hazard = stacked_model.predict(stacked_predictions_test)
    result.to_csv('output.csv', index=False)
