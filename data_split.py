## Data split
    #This contains a function that takes in a data frame and splits the data into training/test and even to validation set


import numpy as np
import pandas as pd
import time, datetime
import sklearn as sk
from sklearn.model_selection import train_test_split
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')


#this function takes in Y and X data sets and split them to three sets: train/validation and test
def modeldata_split_train_test_validation(df_in_X, df_in_y, test_size, validation_size):
    '''
    takes in Y and X data sets and split them to three sets
    Parameters: 
    df_in_X, df_in_y, test_size, validation_size
    return X_train, X_test, X_val,  y_train, y_test, y_val
    '''
    start_time          = time.time()    
    val_adjusted_size=float(validation_size)/(1.0-float(validation_size))
    X_train, X_test, y_train, y_test = train_test_split(df_in_X, df_in_y, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_adjusted_size, random_state=1)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_split.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the split train test validation function is "+ str(time_end) + "\n" + "Shape of Train set is "+ str(X_train.shape)+ "\n" + "Shape of Test set is "+ str(X_test.shape) + "\n" + "Shape of Validation set is "+ str(X_val.shape)+ "\n" + "Shape of Target variable is "+ str(y_train.shape))
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )    
    return X_train, X_test, X_val,  y_train, y_test, y_val

#This takes X, y data sets and split the data to train and test
def modeldata_split_train_test(df_in_X, df_in_y, test_size):
    '''
    takes in Y and X data sets and split them to two sets
    Parameters: 
    df_in_X, df_in_y, test_size
    return X_train, X_test, X_val,  y_train, y_test, y_val
    '''
    start_time          = time.time()  
    X_train, X_test, y_train, y_test = train_test_split(df_in_X, df_in_y, test_size=test_size, random_state=1)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_split.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the split train test function is "+ str(time_end) + "\n" + "Shape of Train set is "+ str(X_train.shape)+ "\n" + "Shape of Test set is "+ str(X_test.shape) + "\n"  + "Shape of Target variable is "+ str(y_train.shape))
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )   
    return X_train, X_test, y_train, y_test
  