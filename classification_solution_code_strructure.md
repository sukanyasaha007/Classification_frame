
# Code Structure Overview

    This folder can be utilized to store all python scripting files from this framework. 
    All the files should be python scripts only (with .py extension) or Ipython notebook.
    The contents of this folder should be checked in to bitbucket.

    The entire pipeline has three main components. 
       1. The actual model related codes which is divided in to three groups by itself (Machine learning models, statistical models and deep learning models)
       2. utility codes (includes reading_data, plot, variable selection, model, parameter configuration, etc) 
       3. finally the part in which everything integrated to work together (main.py file) and  Ipython notebooks

    Below, brief description is given about each file.

## Read Data

    •Reading different file types – It can read csv, parquet, hdf, json, feather, Data file, Database file, SQL database file, Linux / Unix tarball file archive, txt file in a single function
    •Reading multiple files into single DF-  it can multiple parquet files and convert to a single dataframe

## Data Summary
    • Overall summary-     summary details about Count of row, columns, no of numeric variables, no of categorical variables etc for entire data set
    • Null Count Percentage- Percentage of missing value in each variable for entire data set
    • Count of 0/1 Percentage-  Percentage of zero in all column and  Percentage of  one in each numerical column
    • Most Frequent Values- most frequent values in a column
    • Write summary to excel – Creates an excel of all above summary



## EDA
    • Histogram-  Plots histogram of a numerical column, normalizes before plotting if parameter normalize if true.
    • Scatter plots- Scatter plot with respect to x and y
    • Missing Value Plot- Histogram of missing percentage of features
    • Plot unique- Histogram of percentage of unique values
    • Plot bar categorical – Bar plot for categorical variable
    • Plot correlation- Heatmap of the features with correlations above the correlated threshold in the data.
    • Plot summary - plots summary details about Count of row, columns, no of numeric variables, no of categorical variables, number of missing value and unique value for entire data set
    • Plot variable importance- Plots features based on their ranks per some algorithm
    

## Data Preparation
    This includes: 
    • Missing Value Imputation- Removing columns whose null values are above 10% and after that it replaces the missing values with median, most frequent, mean.  
    • Compare Features across Data Frames- It then compares train and test samples and filters them with common columns. The sum of three columns is added as one label column and removed the individual columns.  
    • Outlier Treatment- Takes a data frame a replaces outliers with 5 and 95(which can be passed through parameters) percentile for each numeric column
    • Data Type Casting- drops categorical column which has more than certain number of unique levels and if any numerical column has less than some threshold value of unique entries then the numerical column is converted to categorical column
    • Dimensionality Reduction - Based on Null %
    • Dimensionality Reduction - Only One Level
    • Dimensionality Reduction - Based on Correlation
    • Dimensionality Reduction – through VIF for Multi Collinearity
    • Dimensionality Reduction- through Correlation matrix
    • Encoding labels of Categorical Variables
    • label creation 
    • Standardization- Standardization of numerical variables
    • Feature Matrix Creation- Creates matrix=x of features removing dependent variables present in the data frame


## Directory Path
    This file represents home, data, project directories as variable
    
## Log output
    This file keeps record of performances measures of each algorithm. It writes its outputs in a file called "log_results.txt" file save it in mlresults directory

## config parameters
    This file contains list of parameters used in the whole process. The parameters are parameters that are not used more than once place in the process. 

## Feature Engineering
    • Log Transformations – Log Transformation of a column
    • Reciprocal Transformations- Reciprocal Transformation of a column
    • Square Transformations- Square Transformation of a column
    • Find skewness- gives details of skewness with respect to features
    • Transform Skewed Data- Transforms the skewed variables per Transformation parameter 

## Feature Selection
    • Select k best - Selects k columns based on a parameter
    • Feature ranking-  gives feature rankings for all columns
    • Feature Selection- Selects columns based feature rankings received from algorithm
    
## Data split
    This contains a function that takes in a data frame and splits the data into training/test and even to validation set


## ML Models
    This contains below algorithms and respective grid search function
    • Logistic Regression
    • XG Boost
    • Decision Tree
    • Random Forest
    • Support vector machine
    • Ada boost
    • Catboost
    • light gradient Boost


##  DL models
    This contains multi layer perceptron and neural net algorithm using keras 

## main
    In this file all the necessary functions from other files gets called. This includes reading data, preprocessing, variable selection,  data_split, modleing, etc. 

## H2O Models
    This file contains h2o models.
    • Super learner - Generate a 2-model ensemble (GBM + RF)
    • Deep Learning using h2o

## Model Evaluation
     All the custom created evaluation functions are found in this file
    • accuracy score
    • balanced accuracy score
    • f1 score
    • precision score
    • recall score
    • Log loss
    • roc curve
    • roc auc score
    • confusion matrix
    
## Model Interpretation
    Lime has been used to compare the effect of features on some prediction based on an algorithm


 

