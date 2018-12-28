
# Instructions on how to run the Classification framework
As it is described in the code structure readme file, this pipeline has various models and utility codes that work together to create a complete pipeline. The following are important points users of this pipeline should pay attention before using it

## Set up
- directoryPath.py and config_param.py should be updated with the right path and identifier in order for the program to run smoothly
- At the moment, output files have prefix of "msrp". In order to avoid confusion and overwriting, one has to change this prefix to something else appropriate. This should be changed in the config_param file under the variable name of "project_identifier".
- If the project runs in different directory, directory paths should be modified accordingly in the "config_param.py" file. File names should also be modified as needed. 
- "msrp_regression.py" contains the different functions that need to run (discussed below in detail) and hence it is possible to uncomment and run whatever is needed in the file or alternatively import one of the functions either from the modeling files or other utility files.
- If this pipeline runs on the server, no need to worry about dependencies. Everything is pretty much there. 
- It is important users see the example in the msrp_regression.py file on how to run the code in addition to reading the explanation below.
  
 To run the pipeline end to end, one has to execute the command: 
            "Python msrp_regression.py" 
which calls a number of functions to accomplish various tasks from the different utility files. In other words all the functions are called inside the "Python msrp_regression.py". The different files and the functions they contain are described below. The description below is in the order they have to be executed in the msrp file.



##  Reading file

First file to run is the "reading_data.py" to call the data reading function. This function is very versatile to handle different kind of format. The function accepts file names with directory path and gives out pandas dataframe. However, it also spits out error message in case it encounters some unknown file formats to it. Here in this specific case it reads two csv files represented by variables "raw_data_test_csv" and "raw_data_train_val_csv" in the "config_param.py" file. So, if the user want to run this on new data files, these two names need to be changed. In our particular case, we are have train and test set in a separate file and hence needed to do the reading twice. If only one consolidated data is provided, this needs to be done once. The data can be prepared in a different ways depending on the need. 

Since we do the above reading process and the following preparation mentioned in step 2 once, we commented out that part in "Python msrp_regression.py". The user can uncomment this and use it but also can do their own way of preparation. 

## Data summary 
The "data_summary.py" gets called right after reading the file and it takes any a dataframe and gives out a table contains information such as mean, maximum, number of missing values etc for every column. The output is recorded in mlresult directory under the name data_summary.txt. 

## EDA
For Plot histogram of a numerical column, Scatter plot with respect to x and y, bar plots of categorical variables, percentage of unique values and Missing Value. Then plot the summary Plot to get overall idea about the data and save the plots in mlresult directory. This can be run anytime before or after the data preparation.
                                                        
## Data Prep
Then the dataframe from the first step gets processed by calling functions from the "data_prep.py" file. This file included functions:

#### Removing null values
Remove_null function takes two arguments (df and fraction (0-1)) and spits out another df. It first removes columns whose null values are above the fraction value and after that all rows with any number of null values.

#### Feature selection
Feature_selection dunction takes three arguments df, n1 and n2. It removes columns from df whose number of unique values below n1 and above n2. The final output is a dataframe. N1 and n2 are variables and can be changed at anytime.

#### Selecting common features

selecting_common_col function takes two data frames (separate test and train) and keeps in both the common columns. Output is two columns whose columns name are the same. 

Finally, both data sets (test and train), are converted to "HDF5" format because it is faster to re-read them. Their name is train.h5 and test.h5. After this hdf5 formatted files read everytime the pipeline runs. 

It is possible to start the whole process of running this pipeline from this point by ignoring the above process if the data does not require the processes above or already prepared somewhere else in a different way. But it is important, it is cleaned, all numerical, not categorical and no null values.

#### Outlier Treatment
Takes a data frame a replaces outliers with 5 and 95(which can be passed through parameters) percentile for each numeric column
#### Data Type Casting
Drop categorical column which has more than certain number of unique levels and if any numerical column has less than some threshold value of unique entries then the numerical column is converted to categorical column

#### Dimensionality Reduction 
Based on Null %
Remove features with Only one Level
Based on Correlation
Throu Vif for Multi Colinearity

####  Standarization

Scale the numerical columns

#### Encoding 
Encode the labels of Categorical Variables

#### Compare Features across Data Frames
Compare features of train and application set

#### Feature Matrix Creation
Create matrix of features removing dependent variables present in the data frame


## Feature Engineering
Based on the nature of variables perfor Log Transformations,Reciprocal Transformations, Square Transformations. Also find skewness with respect to features and Transform Skewed Data if required.

## Feature Selection
Select best features through algorithms or find the rank of each feature using functions Select k best, Feature Selection which select columns based feature rankings received from algorithm and Feature ranking

## Data spliting

The file "modelData_spliy.py" contains two functions: "modeldata_split_train_test_validation" and "modeldata_split_train_test". Normally, these two functions can be called to do data splitting into train/test/validation and train/test set respectively. In this particular case, we called the second function to split the training.h5 set only into two sets (train and validation) since we have a standalone test.h5 set. At this point, we have train, validation and test set. In situation where a standalone test data is not provided, the first function should be called to split the data into train/test/validation data. The feature and the label (df_X and df_y) are provided to the splitting function separate not as one frame.

## Configuration parameters

"config_param.py" file contains various variable values that have the same values throughout the pipeline such as train/test split values and file names. When parameters are used with different values in the pipelines at different cases, they are hard coded. Model parameters for example are not part of this file.  

## Modeling

At this stage, the actual modeling codes are executed with three separate categories each with the same set of similar structure in terms of functionalities. The three categories are: Statistical ("StatisticalModels.py"), machine learning ("MLModels.py") and deep learning ("DLModels.py"), ("H2Omodels.py") models. H2O model category to be included in the future

Each category contains several kinds of models of their kind for which we have both baseline and gridsearch-parameter-tuned model codes. The baseline models are pretty much very basic as their name implies while the gridsearch-parameter-tuned models are the best performing ones selected from a host of models with different parameter combination. The user can play around by including and removing the different parameters and varying their values. In all cases, functions to record performance measure (R-square, adjusted R-square and rms), duration of time to run the code from the "log_output.py" file are called. The result is logged in msrp_log_result.txt file in mlresults directory. Furthermore, a residual plot of the result is saved in mlreslt files under the name msrp_plot_residual and model name appended to it. Depending on what evaluation metrics are desired one can comment and uncomment any metric code line in the  "Python msro_regression.py" file.       


In order to run any algorithm and print out the result on terminal, one has to run for example:

print (lasso_grid_search(X_train, y_train, X_test, y_test, True, 1d_df))  

to execute gridsearch for lasso regression and also run lime (The last two arguments are for lime). If one however does not want to run lime, it is possible to run with the default values of the last two arguments as False and None like: 

print (lasso_grid_search(X_train, y_train, X_test, y_test))

If the user does not want to print results on terminal, the above commands can be run without print.

## Lime: model explanation
   Lime is implemented to explain the models. In order to make use of this capability, one has to set the "lime_flag" argument in each of the modeling arguments to true and provide 1-d array whose length is similar to the number of feature of the training data sample. The default for the Lime_flag is false. The output gets recorded in mlresult directory as "msrp_log_lime_explainer.txt" file. 
   
## Model Evaluation
Evaluate models by 
    • Root Mean Squared Error
    • R squared Score
    • Adjusted R Squared Score
and Plot Residuals - present in plot_residuals.py file
              
