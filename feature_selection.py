## Feature Selection
    #• Select k best - Selects k columns based on a parameter
    #• Feature ranking-  gives feature rankings for all columns
    #• Feature Selection- Selects columns based feature rankings received from algorithm

import lightgbm
import sys
import numpy as np

import time
import sys
import warnings
import datetime
import gc
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import os
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm


import plotly as pl


from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn import linear_model
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier, XGBClassifierBase
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, fbeta_score, classification_report
from sklearn.metrics import log_loss, recall_score, roc_auc_score, roc_curve

from sklearn.linear_model import PassiveAggressiveClassifier, RandomizedLogisticRegression

from sklearn.model_selection import GridSearchCV
import lightgbm
import config_param
import data_prep 
import data_summary
import directoryPath
#import evaluation_metric
import lime_ModelExplainer
import log_outputs
import data_split
#import plot_residual
import reading_data
import eda

#from evaluation_metric import adjusted_R2score_calc #custom made
#from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer

project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')

#%matplotlib inline  
import time, datetime
import sklearn
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, chi2, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
ts = time.time()


sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def feature_ranking(X,y, estimator='lin'):
    '''
    gives feature rankings for all columns
    parameter
    -------
    X: Data frame of Indepenedent columns where categorical columns are encoded
    y: Target Variable
    estimator: Algorithm to measure rank of the features. Options-
    lin = Linear Regression
    lasso= Lasso
    ridge= Ridge
    svm= Support Vector Machine
    dt= Decision Tree Regressor
    rf= Random Forest Regressor
    boost= Ada Boost Regressor
    
    '''
    start_time          = time.time()
    # Create the RFE object and rank each feature
    lin=sklearn.linear_model.LinearRegression(n_jobs=-1)
    lasso=sklearn.linear_model.Lasso()
    ridge=sklearn.linear_model.Ridge()
    svm=sklearn.svm.SVR()
    dt=sklearn.tree.DecisionTreeRegressor()
    rf=sklearn.ensemble.RandomForestRegressor()
    boost=sklearn.ensemble.AdaBoostRegressor()
    
    estimator_dict= {'lin': lin, 'ridge': ridge, 'svm': svm, 'dt': dt, 'rf': rf, 'boost':boost} 
    
    #Ranking
    rfe = RFE(estimator_dict[estimator])
    rfe.fit(X, y)
    ranking = rfe.ranking_
    rank_df=pd.Series(ranking, index=X.columns).sort_values()
    #rank_df.name='Rank'
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_feature_selection.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the feature ranking function is "+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return  rank_df
    
def feature_selection(X,y, estimator='logistic', threshold='1.5*median'):
    '''
    Selects columns based feature rankings recieved from algotithm
    parameter
    -------
    X: Data frame of Indepenedent columns where categorical columns are encoded
    y: Target Variable
    estimator: Algorithm to measure rank of the features. Options-
    lin = Linear Regression
    lasso= Lasso
    ridge= Ridge
    svm= Support Vector Machine
    dt= Decision Tree Regressor
    rf= Random Forest Regressor
    boost= Ada Boost Regressor
    lgb= Light gradient Boost
    xgb= Extreem Gradient Boosting
    gb= Gradient Boosting
    
    '''
    start_time          = time.time()
    # Create the RFE object and rank each feature
    log=sklearn.linear_model.LogisticRegression(n_jobs=-1)

    svm=sklearn.svm.SVC()
    dt=sklearn.tree.DecisionTreeClassifier()
    rf=sklearn.ensemble.RandomForestClassifier()
    boost=sklearn.ensemble.AdaBoostClassifier()
    model_lgb = lightgbm.LGBMClassifier(objective='binary',boosting_type='gbdt', num_leaves=31, max_depth=-1, 
                                        learning_rate=0.1, n_estimators=100, 
                                        subsample_for_bin=200000, class_weight=None, 
                                        min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, 
                                        subsample=1.0, subsample_freq=0, colsample_bytree=1.0, 
                                        reg_alpha=0.0, reg_lambda=0.0, random_state=None, 
                                        n_jobs=-1, silent=True,)
    
    model_xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, 
                              objective='binary:logistic', booster='gbtree', n_jobs=1, 
                              nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                              reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, 
                              seed=None, missing=None)
    GBoost = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
                                        subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                        min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                                        random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
                                        warm_start=False, presort='auto', validation_fraction=0.1, 
                                        n_iter_no_change=None, tol=0.0001)
    
    estimator_dict= {'logistic': log, 'svm': svm, 'dt': dt, 'rf': rf, 'boost':boost,
                     'lgb' : model_lgb, 'xgb': model_xgb, 'gb': GBoost} 
    
    embeded_lr_selector = SelectFromModel(estimator_dict[estimator], threshold)
    '''The threshold value to use for feature selection. Features whose importance is greater or equal are kept while the others are discarded. If “median” (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances. A scaling factor (e.g., “1.25*mean”) may also be used. If None and if the estimator has a parameter penalty set to l1, either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5. Otherwise, “mean” is used by default.'''
    embeded_lr_selector.fit(X, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    
    X_new= X[embeded_lr_feature]
    print(str(len(embeded_lr_feature)), 'selected features')
    time_end            =time.time() - start_time
    '''
    f = open(mlresult_dir +str(project_identifier) +"_log_feature_selection.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the feature selection with algorithm is "+ str(time_end) + "\n" str(len(embeded_lr_feature)) + 'selected features ' + "\nselected features are " + str(embeded_lr_feature))
    f.close()
    '''
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return  embeded_lr_feature, X_new