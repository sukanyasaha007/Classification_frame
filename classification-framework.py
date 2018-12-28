
# ## Importing Packages
from __future__ import print_function
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

import time, datetime
import sklearn
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, chi2, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import lightgbm
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, log_loss, recall_score, roc_curve, roc_auc_score, confusion_matrix,classification_report 
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, fbeta_score, classification_report
from sklearn.metrics import log_loss, recall_score, roc_auc_score, roc_curve,accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import linear_model
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier, XGBClassifierBase
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, fbeta_score, classification_report
from sklearn.metrics import log_loss, recall_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
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
from keras.optimizers import SGD
import tensorflow as tf

# ### Appending path to import h2o

sys.path.append('/datascience/home/ssaha/')
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.regression import H2ORegressionModel
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator


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
from log_outputs import log_data_summary
#from evaluation_metric import adjusted_R2score_calc #custom made
#from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer
import h2oModels, mlModels, dlModels

from config_param import project_identifier 

#get_ipython().run_line_magic('matplotlib', 'inline')
from directoryPath import mlresult_dir

gc.collect()


# # Reading data
df_parquet= reading_data.read_data(path="/datascience/home/ssaha/input/c360_customeradt_lexussegmentation_2012_09_30/")


# ### Creating Dependent Column
#primary_key =  df_parquet["customer_id"]           # Primary Key column names, useful in 
y = df_parquet["dep_purchase_lexus_12mo"]  # service dependent variable 


# ## Overall summary
# 
df_parquet.head()
data_summary.get_overall_summary(df_parquet)
#data_summary.get_missing_value_count(df_parquet).head()
#data_summary.get_most_frequent_count(df_parquet).head()
data_summary.write_to_excel(df_parquet)


# # Preprocessing

# ### Remove zero varience
df_model2_removed_one=data_prep.remove_cols_with_one_unique_value(df_parquet)
X= data_prep.find_indep_feat(df_model2_removed_one)
#X.shape
#plt.plot(y.value_counts())
"""
def bar_chart(X,y, feature):
    yes = X.loc[y==1][feature].value_counts()
    no = X.loc[y==0][feature].value_counts()
    df = pd.DataFrame([yes, no])
    df.index = ['Yes','No']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
X.loc[y==1]['customer_type'].value_counts()
bar_chart(X,y, 'customer_type')
"""
# ### Missing Value
"""df_=df_parquet.copy()

for cat_col in df_.select_dtypes(include='O').columns:
    if df_[cat_col].isnull().sum()>0:
        print(df_[cat_col].head())
        print("Categorical column "+ str(cat_col) + " is getting imputed by most frequent value- " +str(df_[cat_col].value_counts().index[0]))
        print(df_[cat_col].head())
        df_[cat_col].fillna(df_[cat_col].value_counts().index[0], inplace=True)

df_['deceased_ind'].value_counts().index[0]"""

X_imputed= data_prep.missing_value_imputation(X, 'mean')


# ### Outlier Treatment
X_outlier_treated= data_prep.outlier_treatment_dataframe(X_imputed)


# ### Removing Few variables
X_outlier_treated.drop(['age_dt', 'age_no',  'household_id', 'customer_id'], axis=1,inplace=True, errors=False)


# ### Type Casting
X_casted, num_to_cat= data_prep.type_casting(X_outlier_treated, 5, 5)


# ### Standarization
X_norm= data_prep.standarization(X_casted)


# ### Multicollinearity Treatment with VIF
vif, high_vif= data_prep.variance_inflation_factors(X_norm.select_dtypes(exclude='O'))

#high_vif

high_vif.tolist().remove('const')

X_vif_treated= X_norm.drop(high_vif, axis=1,errors=False)


#X_vif_treated.shape


# ### Correlation

record_corr, X_corr_treated= data_prep.remove_col_with_corr(X_vif_treated, .7)


X_corr_treated.shape


# ## Encoding

X_corr_treated_encoded= data_prep.label_encode(X_corr_treated)


# ### Treating negative columns
X_corr_treated_encoded_abs=X_corr_treated_encoded.abs()


# ## Storing the processed dataframe
#X_corr_treated_encoded_abs.shape
#X_corr_treated_encoded_abs.dtypes
#y.dtype


# ## Save Features and target
X_h5=X_corr_treated_encoded_abs.to_hdf(mlresult_dir + str(project_identifier) + '_' + str(datetime.datetime.now().day)+ '_X.h5', key= 'df')
y_h5=y.to_hdf(mlresult_dir + str(project_identifier) + '_' + str(datetime.datetime.now().day)+ '_y.h5', key= 'df')

X_h5= pd.read_hdf(mlresult_dir + str(project_identifier) + '_' + str(datetime.datetime.now().day)+ '_X.h5')
y_h5= pd.read_hdf(mlresult_dir + str(project_identifier) + '_' + str(datetime.datetime.now().day)+ '_y.h5')

X_h5.head()
X_h5.shape
type(X_h5)

y= df_parquet['dep_purchase_lexus_12mo']
y.shape
y.head()


# ### Data Split
df= pd.concat([X_h5.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
df.head(15)
sample_df=df.sample(frac= .001)
sample_df.shape
X_train, X_test, y_train, y_test=train_test_split(sample_df.iloc[:, :-1], sample_df['dep_purchase_lexus_12mo'], stratify= sample_df['dep_purchase_lexus_12mo'])



ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

log= mlModels.logistic_regression(X_train, y_train, X_test, y_test, True)


# # Stochastic gredient Descent

sgd= mlModels.sgd_regression(X_train, y_train, X_test, y_test, lime_flag=True)


# # Gaussian Process Classifier

gpc= mlModels.gpc(X_train, y_train, X_test, y_test, lime_flag=True)


# # Decision Tree


dt= mlModels.dt(X_train, y_train, X_test, y_test, lime_flag= True)


# # Random Forest

rf= mlModels.rf(X_train, y_train, X_test, y_test, lime_flag= True)


# # Extreme Gradient Boosting

xgb= mlModels.xgb(X_train, y_train, X_test, y_test, lime_flag= False)


# # Sklearn Gradient Boosting

gb= mlModels.gb(X_train, y_train, X_test, y_test, lime_flag=True)

# # Bagging

bag= mlModels.bagging(X_train, y_train, X_test, y_test, lime_flag=True)


# # Adaboost

adaboost= mlModels.adaboost(X_train, y_train, X_test, y_test, lime_flag= True)


# # Grid Search

grid= mlModels.grid_search(X_train, y_train, X_test, y_test, lime_flag= True,
                tuned_parameters = {'min_samples_split': [2,4,6,8,10], 'min_samples_leaf': [1,2,3,4,5],
                     'min_weight_fraction_leaf': [0, .1, .2], 'max_features': [80,90,100,125,150,175], 
                                    'max_leaf_nodes': [10,20,30,40,50,70,80]},
                model= DecisionTreeClassifier())
 
# # H2O
h2o.init()
train= pd.concat([X_train, y_train], axis=1)
test= pd.concat([X_test, y_test], axis=1)
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)

#x=train.columns

# Deep Learning with h2o
dl_fit, pred= h2oModels.h2o_deeplearning(train,test, "dep_purchase_lexus_12mo", model_id= "h2o_classification")


# Super learner using rf and gbm


my_gbm, my_rf, ensemble, pred= h2oModels.super_learner(train,test, "dep_purchase_lexus_12mo", model_id= "super_learner_classification")


# # Deep Learning with Keras


estimator , y_pred = dlModels.dlModel_classifier(X_train, y_train, X_test, y_test)

# # Multi-layer Perceptron

mlpc= dlModels.mlpc(X_train, y_train, X_test, y_test, lime_flag=True)


# # Feature Selection

#X_h5= pd.read_hdf(str(mlresult_dir)+ "c360_customeradt_lexussegmentation_2012_09_30_10_X.h5")
#X_train,X_test, y_train, y_test = train_test_split(X_h5, y)

## Feature Selection
    #• Select k best - Selects k columns based on a parameter
    #• Feature ranking-  gives feature rankings for all columns
    #• Feature Selection- Selects columns based feature rankings received from algorithm


embeded_lr_feature, X_new=  feature_selection.feature_selection(X_h5, y)

print(embeded_lr_feature)


#pd.set_option('display.max_rows', 500)

#rank_df[rank_df<=1].index.tolist()
#X_new= X_h5[embeded_lr_feature]
#X_new.shape
#y.shape

df= pd.concat([X_new.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
sample_df= df.sample(frac=.01)
y.head()

sample_df["dep_purchase_lexus_12mo"]

X_train,X_test, y_train, y_test = train_test_split(sample_df.drop("dep_purchase_lexus_12mo", axis=1),sample_df["dep_purchase_lexus_12mo"])
#X_train.shape
#y_train.shape
#X_test.shape
#y_test.shape

#Training on Selected features
log= mlModels.logistic_regression(X_train, y_train, X_test, y_test)

sgd=mlModels.sgd(X_train, y_train, X_test, y_test, loss= 'log')


gcp= mlModels.gpc(X_train, y_train, X_test, y_test)

dt= mlModels.dt(X_train, y_train, X_test, y_test)

rf= mlModels.rf(X_train, y_train, X_test, y_test, criterion='entropy', n_estimators=150)

xgb= mlModels.xgb(X_train, y_train, X_test, y_test)

gb= mlModels.gb(X_train, y_train, X_test, y_test)

ada=mlModels.mcp(X_train, y_train, X_test, y_test)





def plot_2d_space(X, y, label='Classes'):   
    pca = PCA(n_components=2)
    X_pca = pd.DataFrame(pca.fit_transform(X))
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter( 
            X_pca.loc[y.reset_index(drop=True)== l, 0],
            X_pca.loc[y.reset_index(drop=True)== l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

plot_2d_space(X_train, y_train, 'Imbalanced training dataset (2 PCA components)')
plot_2d_space(X_test, y_test, 'Imbalanced test dataset (2 PCA components)')


plot_2d_space(X_h5, y, 'Imbalanced dataset (2 PCA components)')


import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


#Raw over- and under- sampling
#A group of researchers implemented the full suite of modern data sampling techniques with the 
#imbalance-learn contrib module for sklearn. This submodule is installed as part of the base sklearn 
#install by default, so it should be available to everyone. It comes with its own documentation as well; 
#that is available here.

#imblearn implements over-sampling and under-sampling using dedicated classes.

rus, id_rus, X_rus, y_rus= data_prep.random_undersample(X_h5, y)


X_ros, y_ros, ros= data_prep.random_over_sample(X_h5, y)


# # Under-sampling: Tomek links
# Tomek links are pairs of very close instances, but of opposite classes. Removing the instances of the majority class of each pair increases the space between the two classes, facilitating the classification process.

X_tl, y_tl, tl, id_tl= data_prep.undersample_tomek_link(X, y, plot=True)

# # Under-sampling: Cluster Centroids
# This technique performs under-sampling by generating centroids based on clustering methods. The data will be previously grouped by similarity, in order to preserve information.

X_cc, y_cc, cc= data_prep.undersample_cluster_centroid(X,y, plot=True)


# # Over-sampling: SMOTE
# SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.
# We'll use ratio='minority' to resample the minority class.

X_sm, y_sm, smote= data_prep.oversample_SMOTE(X,y, plot=True)


# # Over-sampling followed by under-sampling
# Now, we will do a combination of over-sampling and under-sampling, using the SMOTE and Tomek links techniques:

X_smt, y_smt, smt= data_prep.over_under_SMOTETomek(X, y, plot=True)

########## Learning Curve ########

f1s,accs,precs,recalls, sampler, model= data_prep.learning_curve(X_train, X_test,y_train, y_test)

