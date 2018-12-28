#Multi-layer Perceptron


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
from keras.optimizers import SGD

import plotly as pl

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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, fbeta_score, classification_report
from sklearn.metrics import log_loss, recall_score, roc_auc_score, roc_curve

from sklearn.linear_model import PassiveAggressiveClassifier, RandomizedLogisticRegression

from sklearn.model_selection import GridSearchCV


import data_prep 
import data_summary

#import evaluation_metric
import lime_ModelExplainer

import data_split
#import plot_residual
import reading_data
import eda

#from evaluation_metric import adjusted_R2score_calc #custom made
#from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer

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

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
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
from keras.optimizers import SGD
#from evaluation_metric import adjusted_R2score_calc #custom made
#from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer
sys.path.append('/datascience/home/ssaha/')
project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')

    #Multi-layer Perceptron
def mlpc(X_train, y_train, X_test, y_test, lime_flag=False,
       hidden_layer_sizes=(100,), activation='relu', 
                      solver='adam', alpha=0.0001, batch_size='auto', 
                      learning_rate='constant', learning_rate_init=0.001, 
                      power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
                      tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                      nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                      beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10):
    
    '''
    Parameters:
    X_train, y_train, X_test, y_test- Learning set
    lime_flag-  enable or disable lime
    '''
    start_time          = time.time()
    # cretae instance
    mlpc= MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                      solver=solver, alpha=alpha, batch_size=batch_size, 
                      learning_rate='constant', learning_rate_init=0.001, 
                      power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, 
                      tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, 
                      nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, 
                      beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

    mlpc.fit(X_train,y_train)
    #Predict on test set
    y_pred= mlpc.predict(X_test)

    # understand the model through lime
    if lime_flag:
        lime_explainer(X_train, y_train, X_test, y_test, df_row=2,  model_predictor= mlpc, alogorithm_name="mlpc")                                                
    time_end=time.time() - start_time
    # Scores
    model_evaluation(X_train,y_train, X_test, y_test,y_pred, mlpc, time_end, alg_name='mlpc') 
    # resturn model object
    return mlpc
# Using Keras



#~~~~~~~~~~~~~~~~base model~~~~~~~~~~~~~~~~~~~~
def DLmodel_baseline(X_train, y_train, X_test, y_test,loss='binary_crossentropy', 
                     metrics=['accuracy'], 
                     optimizer = tf.train.RMSPropOptimizer(0.001)):
# create model
    model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    model.fit(X_train, y_train, 
               batch_size = 60, 
               epochs = 30, verbose=1)
    return model


# Deep Learning model classifier

def dlModel_classifier(X_train, y_train, X_test, y_test,EPOCHS = 100, batch_size=5,
                        loss='binary_crossentropy', metrics=['accuracy'],
                        optimizer = SGD(lr = 0.01, momentum = 0.9)):
    start_time = time.time()
    estimator = DLmodel_baseline(X_train,y_train, X_test, y_test, loss, metrics, optimizer)
    # Validation: ¶
    # Fit the model
    history = estimator.fit(X_train, y_train, validation_split=0.20, 
                    epochs=180, batch_size=10, verbose=0)

    # list all data in history
    print(history.history.keys())
    #estimator.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=batch_size) 

    y_pred= estimator.predict_classes(X_test)
    
    # summarizing historical accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    time_end =time.time() - start_time
    

    # Scores
    acc =accuracy_score(y_pred= y_pred, y_true=y_test) * 100
    print('accuracy '+ str(acc))
    print('balanced_accuracy_score '+ str(balanced_accuracy_score(y_pred= y_pred, y_true=y_test) * 100))

    #f1_score
    f1=f1_score(y_pred= y_pred, y_true=y_test, average='macro')  * 100
    print('f1_score ' + str(f1))
    #precision_score
    prec=precision_score(y_pred= y_pred, y_true=y_test, average='weighted')* 100
    print('precision_score ' + str(prec))
    #log_loss
    print('Log loss ' + str(log_loss(y_pred= y_pred, y_true=y_test) ))
    #recall_score
    recall=recall_score(y_pred= y_pred, y_true=y_test)*100
    print('recall_score ' + str( recall))
    #roc_curve
    y_pred_proba = estimator.predict_proba(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_test, y_score=y_pred_proba)

    plt.plot([0,1],[0,1],'k--')
    plt.plot(false_positive_rate,true_positive_rate, label="Keras")
    plt.xlabel('false_positive_rate')
    plt.ylabel('true_positive_rate')
    plt.title('ROC curve of ' + "Keras")
    plt.show()

    #roc_auc_score
    roc= roc_auc_score(y_test,y_pred_proba) * 100
    print('roc_auc_score ' + str(roc))

    #confusion_matrix
    print('confusion_matrix ' + str(confusion_matrix(y_test,y_pred)))

    class_report= classification_report(y_test,y_pred)
    print("classification_report"+ str(class_report))
    f = open(mlresult_dir +str(project_identifier) +"_log_dlModels.csv","a")

    f.write("\n Time taken to execute the Keras is "+
            str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now()) +"\n" +
            ' Accuracy Score'+ str(acc) +"\n"+'f1_score ' + str(f1)+ "\n"+'precision_score ' + 
            str(prec)+"\n"+ 'recall_score ' + str( recall) +"\n"+'roc_auc_score ' + str(roc)+"\n" +
            'classification_report '+ str(class_report))
    f.close()
    print("\n Time taken to execute the Keras is " + str(time_end))

    print("\n" +"Dated on"+ str(datetime.datetime.now())+ "\n")
    # resturn model object

    return estimator , y_pred