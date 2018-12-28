###Models
# logistic regression
# Stochastic Gradient Descent
#GaussianProcessClassifier
# Decision Tree
# Random Forest
# Extreme Gradient boosting
#Gradient boosting
#adaboost


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
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, BaggingClassifier, AdaBoostClassifier



from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, log_loss, recall_score, roc_curve, roc_auc_score, confusion_matrix,classification_report 
from sklearn.metrics import  adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import fbeta_score

from sklearn.linear_model import PassiveAggressiveClassifier, RandomizedLogisticRegression

from sklearn.model_selection import GridSearchCV

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
import model_evaluation
from model_evaluation import model_evaluation

from lime_ModelExplainer import lime_explainer

project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')
  
    # logistic regression
def logistic_regression(X_train, y_train, X_test, y_test, lime_flag=False, penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=-1):
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    penalty,  dual, tol=, C=, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose=, warm_start, n_jobs : Parameters to sklearn logistic regression class
    '''
    start_time          = time.time()
    # cretae instance
    log= LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter,  verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
    #fit on train set
    log.fit(X_train,y_train)
    #Predict on test set
    y_pred= log.predict(X_test)
    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= log, alogorithm_name="Logistic_Regression")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, log, time_end, alg_name='Logistic_Regression') 
    # resturn model object
    return log


    # Stochastic Gradient Descent
def sgd(X_train, y_train, X_test, y_test, lime_flag=False, 
                   loss= 'log', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                   fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0, 
                   epsilon=0.1, n_jobs=-1, random_state=42, learning_rate='optimal', 
                   eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, 
                   n_iter_no_change=5, class_weight=None, warm_start=False, average=False, n_iter=None):
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    sgd= SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, 
                   fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, 
                   epsilon=epsilon, n_jobs=n_jobs, random_state=random_state, learning_rate=learning_rate, 
                   eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, 
                   n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start, average=average,
                       n_iter=n_iter)
    sgd.fit(X_train,y_train)
    #Predict on test set
    y_pred= sgd.predict(X_test)
    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= sgd, alogorithm_name="SGD")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, sgd, time_end, alg_name='SGD') 
    # resturn model object
    return sgd 


    #GaussianProcessClassifier
def gpc(X_train, y_train, X_test, y_test, lime_flag=False, kernel=1.0 * RBF(1.0),
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=0,warm_start=False ,
        random_state=42, n_jobs=-1 ,
        max_iter_predict= 1000 ,
        copy_X_train=True ):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    gpc= GaussianProcessClassifier(kernel=1.0 * RBF(1.0) ,
                                   optimizer=optimizer, 
                                   n_restarts_optimizer=n_restarts_optimizer,
                                   max_iter_predict=max_iter_predict ,
                                   warm_start=warm_start, 
                                   copy_X_train=copy_X_train,
                                   random_state=random_state,
                                   n_jobs=n_jobs)

    gpc.fit(X_train,y_train)
    #Predict on test set
    y_pred= gpc.predict(X_test)
    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= gpc, alogorithm_name="gpc")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, gpc, time_end, alg_name='gpc') 
    # resturn model object
    return gpc


    # Decision Tree
def dt(X_train, y_train, X_test, y_test, lime_flag=False,
      criterion='gini', splitter='best', 
                               max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                               class_weight=None, presort=False):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    dt= DecisionTreeClassifier(criterion=criterion, splitter=splitter, 
                               max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                               min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, 
                               max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, 
                               class_weight=class_weight, presort=presort)

    dt.fit(X_train,y_train)
    #Predict on test set
    y_pred= dt.predict(X_test)
    feat_imp= pd.DataFrame(dt.feature_importances_,index = X_train.columns,
                                       columns=['Importance']).sort_values('Importance',ascending=False)
    print(feat_imp.loc[feat_imp['Importance']>0].shape[0] , "Features have more than zero importance")
    print(feat_imp.loc[feat_imp['Importance']>0])
    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= dt, alogorithm_name="Decision Tree")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, dt, time_end, alg_name='Decision Tree') 
    # resturn model object
    return dt


    # Random Forest
def rf(X_train, y_train, X_test, y_test, lime_flag=False,
                      n_estimators=50, criterion='gini', max_depth=None, 
                       min_samples_split=40, min_samples_leaf=20, min_weight_fraction_leaf=0.0, 
                       max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                       min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
                       random_state=42, verbose=0, warm_start=False, class_weight=None):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    rf= RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                               min_weight_fraction_leaf=min_weight_fraction_leaf, 
                       max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, 
                       min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, 
                       random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)

    rf.fit(X_train,y_train)
    #Predict on test set
    y_pred= rf.predict(X_test)

    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= rf, alogorithm_name="rf")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, rf, time_end, alg_name='rf') 
    # resturn model object
    return rf


    # Extreme Gradient boosting
def xgb(X_train, y_train, X_test, y_test, lime_flag=False,
                      max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, 
       objective='binary:logistic', booster='gbtree', n_jobs=-1, nthread=None, gamma=0, 
       min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
       colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
       base_score=0.5, random_state=42, seed=None, missing=0):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    xgb= XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, silent=silent, 
       objective=objective, booster=booster, n_jobs=n_jobs, nthread=nthread, gamma=gamma, 
       min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample, colsample_bytree=colsample_bytree, 
       colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, 
       base_score=base_score, random_state=random_state, seed=seed, missing=missing)

    xgb.fit(X_train,y_train)
    #Predict on test set
    y_pred= xgb.predict(X_test)

    # understand the model through lime
    #if lime_flag:
    #    lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= xgb, alogorithm_name="XGB")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, xgb, time_end, alg_name='XGB') 
    # resturn model object
    return xgb
    
    
    #Gradient boosting
def gb(X_train, y_train, X_test, y_test, lime_flag=False,
       loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
                                    min_impurity_split=None, init=None, random_state=None, max_features=None, 
                                    verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', 
                                    validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    gb= GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, 
                                    criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, presort=presort, 
                                    validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol)

    gb.fit(X_train,y_train)
    #Predict on test set
    y_pred= gb.predict(X_test)

    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= gb, alogorithm_name="gb")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, gb, time_end, alg_name='gb') 
    # resturn model object
    return gb


    #adaboost
def adaboost(X_train, y_train, X_test, y_test, lime_flag=False,
       base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    adaboost= AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, 
                            algorithm=algorithm, random_state=random_state)

    adaboost.fit(X_train,y_train)
    #Predict on test set
    y_pred= adaboost.predict(X_test)

    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= adaboost, alogorithm_name="adaboost")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, adaboost, time_end, alg_name='adaboost') 
    # resturn model object
    return adaboost
    # Grid Search
def grid_search(X_train, y_train, X_test, y_test, lime_flag= True,
                tuned_parameters = [{'kernel': ['rbf'] , 'gamma': [1e-3, 1e-4] , 
                     'C': [1, 10, 100, 1000]} , {'kernel': ['linear'] , 'C': [1, 10, 100, 1000]}] ,model= SVC()):
    '''
    X_train, y_train, X_test, y_test:  train and test set
    tuned_parameters: parameters to be tuned
    model: classifier for grid search 
    '''
    start_time          = time.time()
    
    scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(model, tuned_parameters, cv=5,scoring= score)
        clf.fit(X_train, y_train)
        #Best parameters
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print((mean, std * 2, params))
        print()
        
        #classification report
        print("Detailed classification report:")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        
        # Accuracy Score
        acc =accuracy_score(y_pred= y_pred, y_true=y_test) * 100
        print('Accuracy '+ str(acc))
        print('Balanced accuracy score '+ str(balanced_accuracy_score(y_pred= y_pred, y_true=y_test) * 100))

        #f1_score
        f1=f1_score(y_pred= y_pred, y_true=y_test, average='macro')  * 100
        print('F1 score ' + str(f1))
        
        #precision_score
        prec=precision_score(y_pred= y_pred, y_true=y_test, average='weighted')* 100
        print('Precision score ' + str(prec))
        
        #log_loss
        print('Log loss ' + str(log_loss(y_pred= y_pred, y_true=y_test) ))
        
        #recall_score
        recall=recall_score(y_pred= y_pred, y_true=y_test)*100
        print('Recall score ' + str( recall))
        
        #roc_curve
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_test, y_score=y_pred_proba)

        plt.plot([0,1],[0,1],'k--')
        plt.plot(false_positive_rate,true_positive_rate, label='Grid Search')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve of ' + 'Grid Search')
        plt.show()

        #roc_auc_score
        roc= roc_auc_score(y_test,y_pred_proba) * 100
        print('ROC AUC score ' + str(roc))

        #confusion_matrix
        print('Confusion matrix ' + str(confusion_matrix(y_test,y_pred)))
        pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

        # understand the model through lime
        if lime_flag:
            lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= clf, alogorithm_name="Grid Search")                                                
        time_end=time.time() - start_time
        # Scores
        model_evaluation(X_train,y_train, X_test, y_test,y_pred, clf, time_end, alg_name='Grid Search') 
        # resturn model object
        return clf
