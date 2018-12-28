
## Data Preparation
#This includes: 
    #• Missing Value Imputation- Removing columns whose null values are above 10% and after that it replaces the missing values with median, most frequent, mean.  
    #• Compare Features across Data Frames- It then compares train and test samples and filters them with common columns. The sum of three columns is added as one label column and removed the individual columns.  
    #• Outlier Treatment- Takes a data frame a replaces outliers with 5 and 95(which can be passed through parameters) percentile for each numeric column
    #• Data Type Casting- drops categorical column which has more than certain number of unique levels and if any numerical column has less than some threshold value of unique entries then the numerical column is converted to categorical column
    #• Dimensionality Reduction - Based on Null %
    #• Dimensionality Reduction - Only One Level
    #• Dimensionality Reduction - Based on Correlation
    #• Dimensionality Reduction – through VIF for Multi Collinearity
    #• Dimensionality Reduction- through Correlation matrix
    #• Encoding labels of Categorical Variables
    #• label creation 
    #• Standardization- Standardization of numerical variables
    #• Feature Matrix Creation- Creates matrix=x of features removing dependent variables present in the data frame

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import plotly as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from directoryPath import mlresult_dir
from config_param import project_identifier 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import sklearn
import data_split
project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')

ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def remove_null(df_in, fraction):
    df_1=df_in[df_in.columns[df_in.isnull().mean() < fraction]] #lower limit of missing                                                                                   #values
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a") 
    f.write(sttime + '\n')
    f.write("The original dimension of the data is" +"\t"+ str(df_in.shape) +"\t"+ " and after columns whose missing values are" +"\t"+ str(fraction* 100) +"\t"+ " % and above are removed, the size is " +"\t"+ str(df_1.shape) + "\n" )
    df_out=df_1.dropna()                                        #removing rows with null
    f.close()
    #~~~~~~~~~~~~~~~logging~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir + str(project_identifier) +"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Finally after all rows with one or more missing values are removed , the size is " +"\t"+ str(df_out.shape) + "\n" )
    f.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

    return df_out

#This function selects variables whose number of unique values are between a certain value and also removes #catagorical variables
def feature_selection(df_in, n1, n2):                   #n1=lower bound for unique               
    #lower bound of 
                                                        # unique elements  values and n2 is higher 
    n_uniq_values=df_in.nunique()                       #number of unique values in column
    n_uniq_boolean= n_uniq_values >= n1                 #boolean values of Columns with                                                                           #unique values greater than n1
    df_in=df_in[n_uniq_boolean.index[n_uniq_boolean]]   #Filtering col with the boolean
    
    #upper bound of                                      # unique elements
    n_uniq_values1=df_in.nunique()                       #unique values in column
    n_uniq_boolean1= n_uniq_values1 <= n2                #Columns with unique values                                                                                #greater less than  n2
    df_in=df_in[n_uniq_boolean1.index[n_uniq_boolean1]]  #creating boolean 
    df_out=df_in._get_numeric_data()                     #removes catagorical variables
    
    #~~~~~~~~~~~~~~~~~Logging~~~~~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Keeping only features whose number of unique values are between" +"\t"+str(n1)+ "\t" + str(n2)+"."+"\t"+ "The resulting dataframe has a dimension of"+"\t"+ str(df_out.shape) + "\n" )
    f.close()
    return df_out


#Selectiing the feature intersection of two data frames
def selecting_common_col(df_in1, df_in2):
    if len(df_in1.columns) > len(df_in2.columns): #first data
        col2=df_in2.columns
        df_out1=df_in1[col2]
        df_out2=df_in2
        
    elif len(df_in2.columns) > len(df_in1.columns): #second data
        col1=df_in1.columns
        df_out2=df_in2[col1]
        df_out1=df_in1
        
    else:
        df_out1=df_in1
        df_out2=df_in2
    
    #~~~~~~~~~~~~~~~~~Logging~~~~~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("After filtering the frame with more number of columns with columns of the fame with less number of columns, we have" +"\t"+ str(df_in1)+"\t"+ "and" + "\t" + str(df_in2) + "with dimensions of"+ "\t" +str(df_in1.shape)+"\t"+ "and"+ str(df_in2.shape) + "respevtively"+ "\n" )
    f.close()

    return df_out1, df_out2
        
#creating target column by adding three of teh columns and eventually deleting these three columns. The final result is the features and the label are different.
'''
def creating_label_features(df_in):
    df_y =  df_in['dep_new_msrp_lexus_12mo'] + df_in['dep_cpo_msrp_lexus_12mo'] + df_in['dep_used_msrp_lexus_12mo']
    df_X =  df_in.drop(columns=['dep_new_msrp_lexus_12mo','dep_cpo_msrp_lexus_12mo','dep_used_msrp_lexus_12mo']) #the three features to be combined
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Finally, after summing three columns to get one dependent variable, we end up with df_X and df_y" +"\t"+ "with dimension" + "\t" + str(df_X.shape) +  "and"+ str(df_y.shape) + "respevtively"+ "\n" )
    f.close()
    return df_X, df_y
'''
    
### Adding New functions 11/13/2018

def null_value_counts(df):
    '''
    Takes a data frame and returns count of null values with respect to each feature
    Parameter
    ---------
    df: Dataframe
    Returns: dataframe containing percentage of Counts of null values of each variable
    '''
    start_time          = time.time()
    count_percentage = pd.DataFrame(((df.isnull().sum())/df.shape[0])*100, columns= ['Percentage of missing values'])
    count_percentage['Percentage of missing values'].sort_values(ascending=False)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write("Null Value Counts " +"\n"+str(count_percentage)+ "\n"+ "Time taken to execute the function is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return count_percentage
# Get the list of columns having only one unique value:
def get_cols_with_one_unique_value(df):
    '''
    Takes a data frame and returns list of columns with only one unique value
    Parameter
    ---------
    df: Dataframe
    Returns: list containing independent variable names which have only one unique value
    
    '''
    start_time          = time.time()
    global cols_with_one_unique_val
    cols_with_one_unique_val= []
    for col in df.columns:
        if df[col].nunique()==1:
            cols_with_one_unique_val.append(col)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write("Columns with only one unique value " +"\n"+str(cols_with_one_unique_val)+ "\n"+ "Time taken to execute the function is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return cols_with_one_unique_val

#remove the columns that have only one unique value

def remove_cols_with_one_unique_value(df):
    '''
    Takes a data frame and returns self after removing columns having only one unique value
    Parameter
    ---------
    df: Dataframe
    Returns: Dataframe without columns that have only one unique value
    
    '''
    get_cols_with_one_unique_value(df)
    return df.drop(labels=cols_with_one_unique_val, axis=1)

# Count number of numerical columns
def get_numeric_cols(df):
    '''
    Keeps only the numerical columns
    Parameter: A dataframe
    Returns: Only numeric colums of the dataframe
   
    '''
    return df.select_dtypes(exclude='O')

# Count number of categorical columns
def get_categorical_cols(df):
    '''
    Keeps only the numerical columns
    Parameter: A dataframe
    Returns: Only numeric colums of the dataframe
   
    '''
    return df.select_dtypes(include='O')

# Data Type Casting
def type_casting(df, num_level, cat_lavel):
    '''
    Takes a data Frames and drops categorical column which has more than certain number of unique levels and if any numerical column has less than num_level of unique values then the numerical column is converted to categorical column
    Parameters
    -------
    df: A dataframe 
    cat_level: number of levels, if any categorical column has more than this number of unique levels the categorical column is dropped
    num_level: number of unique values a numeric column must have, if any numerical column has less than these much unique values then the numerical column is converted to categorical column
    
    Returns: Imputed dataframe
    '''
    start_time          = time.time()
    num_to_cat=[]
    count=0
    for col in df.select_dtypes(exclude='object').columns:
        if df[col].nunique()<num_level and not col.startswith('dep'):
            num_to_cat.append(col)
            print("Converting to categorical variable from numerical: " + str(col))
            df[col]= df[col].astype(np.object)
            count+=1
    print('Total number of numerical Variables changed to Categorical Variable is '+ str(count))
    count=0
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique()>cat_lavel and not col.startswith('dep'):
            print('Dropping Categorical Variable ' + str(col))
            df.drop(col, axis=1, inplace=True)
            count+=1
    print('Total number of Categorical Variables dropped is '+ str(count))
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write("Numerical columns converted to categorical columns " +"\n"+str(num_to_cat)+' Total number of Categorical Variables dropped is '+ str(count) + "\n"+ " Time taken to execute the function is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return df, num_to_cat

#Missing value Imputation
"""
def missing_value_imputation(df, method, thresh=50):
    '''
    Takes a data Frames and replaces the missing values with  median, most frequest, mean
    Parameters
    -------
    df: A dataframe 
    method: the method of imputation
    thresh_cat: threshold value or percentage of null rows in a categorical column beyond which the categorical column is dropped
    
    thresh_num: threshold value or percentage of null rows in a numerical column beyond which the numerical column is dropped
    
    Returns: Imputed dataframe
    '''
    df.dropna(how='all', inplace=True, axis=1) #Drops column with all null values
    rows=df.shape[0]
    thresh= np.round((rows*thresh)/100)
    df.dropna(axis=1, thresh=thresh, inplace=True)
    #Impute categorical columns
    #if the column has more than threshold values missing then remove it
    #if its less than 30% values are missung then replace with most frequesnt class
    #if its more than 30% but less than 50% values are missing then create a new class 'NA'
    #get a dataframe of only categorical variables
    #
    #imputer_cat = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
    #imputed_df_cat = pd.DataFrame(imputer_cat.fit_transform(df_cat))
    #imputed_df_cat.columns = df_num.columns
    
    categorical_list = []
    numerical_list = []
    for i in df.columns.tolist():
        if df[i].dtype=='object':
            categorical_list.append(i)
        else:
            numerical_list.append(i)
    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))
    for cat_col in categorical_list:
        null_counts=df_cat[cat_col].isnull().sum()
        if null_counts>0 and null_counts<rows*thresh :
            df[cat_col].fillna(df[cat_col].value_counts().index[0], inplace=True)
       # else :
            #df_cat[cat_col].fillna('NA', inplace=True)
    #df[categorical_list].apply(lambda x: x.fillna(x.value_counts().index[0], inplace=True))

    #Imputation for numeric Columns
    #df_num= get_numeric_cols(df)
        
    #imputation
    fill_NaN = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(X[numerical_list]))
    imputed_df.columns = numerical_list
    result= pd.concat([X[categorical_list], imputed_df], axis=1)
    #df[numerical_list] = Imputer(missing_values='NaN', strategy=method, axis=0).fit_transform(df[numerical_list])
    #imputed_df_num.columns = df_num.columns
    #imputed_df.index = df_num.index
    #for i in df_cat.columns:
     #   imputed_df_num[i]=df_cat[i]
    
    #result= pd.concat([imputed_df_num.sort_index(), df_cat.sort_index()], axis=1)
    return result
"""
def missing_value_imputation(df, method, thresh=50):
    '''
    Takes a data Frames and replaces the missing values with  median, most frequest, mean
    Parameters
    -------
    df: A dataframe 
    method: the method of imputation
    thresh_cat: threshold value or percentage of null rows in a categorical column beyond which the categorical column is dropped
    
    thresh_num: threshold value or percentage of null rows in a numerical column beyond which the numerical column is dropped
    
    Returns: Imputed dataframe
    '''
    start_time          = time.time()
    df.dropna(how='all', inplace=True, axis=1) #Drops column with all null values
    rows=df.shape[0]
    thresh= np.round((rows*thresh)/100)
    df.dropna(axis=1, thresh=thresh, inplace=True)
    #Impute categorical columns
    #if the column has more than threshold values missing then remove it
    #if its less than 30% values are missung then replace with most frequesnt class
    #if its more than 30% but less than 50% values are missing then create a new class 'NA'
    #get a dataframe of only categorical variables
    #
    #imputer_cat = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
    #imputed_df_cat = pd.DataFrame(imputer_cat.fit_transform(df_cat))
    #imputed_df_cat.columns = df_num.columns
    
    categorical_list = df.select_dtypes(include="O").columns.tolist()
    numerical_list = df.select_dtypes(exclude="O").columns.tolist()

    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))
    for cat_col in categorical_list:
        if df[cat_col].isnull().sum()>0:
            print("Categorical column "+ str(cat_col) + " is getting imputed by most frequent value- " +str(df[cat_col].value_counts().index[0]))
            df[cat_col].fillna(df[cat_col].value_counts().index[0], inplace=True)

    fill_NaN = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(df[numerical_list]))
    imputed_df.columns = numerical_list
    #print(df[categorical_list].head())
    #print(imputed_df.head())
    result= pd.concat([df[categorical_list].sort_index(), imputed_df.sort_index()], axis=1)
    #print(result.head())
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the Missing Value Imputation function is "+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return result
#Outlier Detection
# a number "a" from the vector "x" is an outlier if 
# a > median(x)+1.5*iqr(x) or a < median-1.5*iqr(x)
# iqr: interquantile range = third interquantile - first interquantile
"""
def outlier_treatment_dataframe(df):
    '''
    Takes a data frame a replces outliers with 5 and 95 percentile for each numeric column
    Parameter: Dataframe
    Returns: Same dataframe with outliers replaced by 5 and 95 percentile for each numeric column
    '''
    df_num= get_numeric_cols(df)
    df_non_num= get_non_numeric_cols(df)
    icq=df_num.quantile(.75)-df_num.quantile(.25)
    h=df_num.quantile(.75)+1.5*icq
    l=df_num.quantile(.25)-1.5*icq
    df_num.mask(df_num <l, df_num.quantile(.05), axis=1, inplace=True)
    df_num.mask(df_num >h, df_num.quantile(.95), axis=1, inplace=True)
    result=pd.concat([df_num, df_non_num], sort='True', axis=1)
    return result
"""
def outlier_treatment_dataframe(df, iqr_thresh=1.5):
    '''
    Takes a data frame a replces outliers with 5 and 95 percentile for each numeric column
    Parameter: Dataframe
    Returns: Same dataframe with outliers replaced by 5 and 95 percentile for each numeric column
    '''
    start_time = time.time()
    #categorical_list = []
    numerical_list = []
    for i in df.columns.tolist():
        if df[i].dtype!='object':
            numerical_list.append(i)
    icq=df[numerical_list].quantile(.75)-df[numerical_list].quantile(.25)
    h=df[numerical_list].quantile(.75)+iqr_thresh*icq
    l=df[numerical_list].quantile(.25)-iqr_thresh*icq
    df[numerical_list].mask(df[numerical_list] <l, df[numerical_list].quantile(.05), axis=1, inplace=True)
    df[numerical_list].mask(df[numerical_list] >h, df[numerical_list].quantile(.95), axis=1, inplace=True)
    #result=pd.concat([df_num, df_non_num], sort='True', axis=1)
    time_end = time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write( " Time taken to execute outlier treatment function is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return df
def outlier_treatment_vector(x ):
    '''
    Takes a pandas series or dataframe column or any other vector and returns its outliers replaced by 5 and 95 percentile for each numeric column
    Parameter: a vector
    Returns: A vector with its outliers replaced by 5 and 95 percentile for each numeric column
   '''
    icq=x.quantile(.75)-x.quantile(.25)
    h=x.quantile(.75)+1.5*icq
    l=x.quantile(.25)-1.5*icq
    x.replace(x[x <l], x.quantile(.05), inplace=True)
    x.replace(x[x >h], x.quantile(.95), inplace=True)
    return x


def label_encode(df):
    #Encoding labels of categorical variables
    '''
    Takes a data frame and returns the same with all the categorical labels encoded in integer 
    Parameter
    -----
    df: a dataframe
    Returns: encoded self
    '''
    """
    le=LabelEncoder() # create instance of sklearn label encoder method
    df_cat=get_non_numeric_cols(df)
    df_cat_encoded=pd.DataFrame(None)
    for i in df_cat.columns:
        encoded=pd.DataFrame(le.fit_transform(df_cat[i]))
        df_cat_encoded= pd.concat((encoded, df_cat_encoded), axis=1)
    df_cat_encoded.columns=df_cat.columns
    df_encoded=pd.concat((df_cat_encoded, df.select_dtypes(include='number')), axis=1)
    """
    start_time = time.time()
    categorical_list = []
    
    for i in df.columns.tolist():
        if df[i].dtype=='object':
            categorical_list.append(i)

    print('Number of categorical features:', str(len(categorical_list)))
        
    df_encoded=pd.concat((df[categorical_list].apply(lambda x : LabelEncoder().fit_transform(x)), df.select_dtypes(exclude="O")), axis=1)
    time_end = time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write('Number of categorical features:'+ str(len(categorical_list))+ " Time taken to execute outlier treatment function is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return df_encoded



# Correlation 
def remove_col_with_corr(df, correlation_threshold=.7):
    '''
    Takes a data frame and returns the it after removing columns based on correlation threshold
    Parameter:
    -----
    df :a dataframe
    correlation_threshold: threshold value of correlation beyond which features need to be dropped
    Returns: record_collinear and dataframe of features exceeding the correlation threshold
    '''
    start_time = time.time()
    global record_collinear
    corr_matrix = df.corr()
    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
    # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:

    # Find the correlated features
        corr_features=list(upper.index[upper[column].abs() > correlation_threshold])

    # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]    

    # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,'corr_feature': corr_features,'corr_value': corr_values})
    # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index = True)
    df.drop(labels=record_collinear['drop_feature'], axis=1)
    time_end = time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write( " Time taken to execute collinearity treatment function is "+"\t"+ str(time_end) + "\n" + " Collinearity record is "+ str(record_collinear) +"\n")
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return record_collinear, df.drop(labels=record_collinear['drop_feature'], axis=1)

# Collinearity using VIF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def variance_inflation_factors(exog_df, vif_threshold=10):
    '''
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars) only numerical data do not give encoded categorical values
        design matrix with all explanatory variables, as for example used in
        regression.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    start_time = time.time()
    exog_df = sm.add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - sm.OLS(exog_df[col].values, 
                       exog_df.loc[:, exog_df.columns != col].values).fit().rsquared) 
         for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    high_vif=vifs[vifs.sort_values(ascending=False)>vif_threshold].index
    time_end = time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write( " Time taken to execute find VIF function is "+"\t"+ str(time_end) + "\n" + " Following columns are having VIF more than threshold "+ str(vif_threshold)+"\n" + str(high_vif) +"\n")
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return vifs, high_vif
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def calculate_vif_(df, thresh=5):
    '''
    Calculates VIF each feature in a pandas dataframe
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param X: the pandas dataframe
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with features removed
    '''
    start_time = time.time()
    const = add_constant(df)
    cols = const.columns
    variables = np.arange(const.shape[1])
    vif_df = pd.Series([variance_inflation_factor(const.values, i) 
               for i in range(const.shape[1])], 
              index=const.columns).to_frame()

    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df = vif_df.drop('const')
    time_end = time.time() - start_time
    #vif_df = vif_df[vif_df['VIF'] > thresh]
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write( " Time taken to calculate vif is "+"\t"+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to calculate vif is "+ str(time_end) + "\n" )
    #print ('Features above VIF threshold:\n')
    return vif_df[vif_df['VIF'] > thresh]
"""
def remove_collin_with_vif(X, vif_threshold=10):
    '''
    Takes a data frame and returns the columns to remove based on vif threshold
    Parameter:
    -----
    X :a dataframe with encoded categorical columns
    vif_threshold: threshold value of vif beyond which features need to be dropped
    Returns: dataframe of features exceeding the vif threshold
    '''
    
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > vif_threshold:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc) + ' and with vif= ' +                         str(max(vif)))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]
"""
#Remove database name from each variable if the variable name and database name are joined by "."
def remove_database_name(df):
    '''
    Takes a DataFrame and returns variable names if the variabel names were joined with "." to the database name
    Parameter: A dataframe
    Returns: List of column names containing only variable names
    '''
    start_time = time.time()
    var=[]
    for col in df.columns:
        #if i.startswith('d'):
        var.append(col.split('.')[1])
    time_end = time.time() - start_time
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return var
#Find dependent variable names that starts with 'dep'
def find_dep_var(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns list of dependent columns
    Parameter: A dataframe
    Returns: List of dependent variables
    '''
    dep_var=[]
    for col in df.columns:
        if col.startswith('dep'):
            dep_var.append(col)
    return dep_var
#Find matrix of independent variable names when dependent variable names starts with 'dep'
def find_indep_feat(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns dataframe of independent columns
    Parameter: A dataframe
    Returns: A dataframe of independent variables
    '''
    '''indep_var=[]
    for col in df.columns:
        if not col.startswith('dep'):
            indep_var.append(col)
    X=pd.DataFrame(None)
    for i in indep_var:
        X=pd.concat((X, df[i]), axis=1)'''
    return df.drop(list(df.filter(regex= 'dep_')), axis=1)
#Find independent variable names when dependent variable names starts with 'dep'
def find_indep_var(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns list of independent columns' names
    Parameter: A dataframe
    Returns: A list of independent variables
    '''
    #indep_var=[]
    #for col in df.columns:
     #   if not col.startswith('dep'):
    #        indep_var.append(col)
    return df.drop(list(df.filter(regex= 'dep')), axis=1).columns

def standarization(df):
    '''
    Standardization of numerical variables
    Note: first do standarization then missing value imputation
    
    parameter: df- a dataframe
    return: self
    '''
    start_time = time.time()
    categorical_list = []
    numerical_list = []
    for i in df.columns.tolist():
        if df[i].dtype=='object':
            categorical_list.append(i)
        else:
            numerical_list.append(i)
    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))
    #for cat_col in categorical_list:
    scaler = StandardScaler()
    scaler.fit(df[numerical_list])
    result= pd.concat([pd.DataFrame(scaler.transform(df[numerical_list]), columns=df.select_dtypes(exclude="O").columns).sort_index(), df.select_dtypes(include= "O").sort_index() ], axis=1)
    time_end = time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.csv","a")
    f.write(sttime + '\n')
    f.write( " Time taken to execute Standarization function is "+"\t"+ str(time_end) + "\n")
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return result



################# Sampling ###############

######## Use these only if you have imblearn package available
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

#Raw over- and under- sampling
#A group of researchers implemented the full suite of modern data sampling techniques with the 
#imbalance-learn contrib module for sklearn. This submodule is installed as part of the base sklearn 
#install by default, so it should be available to everyone. It comes with its own documentation as well; 
#that is available here.

#imblearn implements over-sampling and under-sampling using dedicated classes.
def random_undersample(X, y, label= "Random UnderSampling", plot=False):
    rus = RandomUnderSampler(return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(X, y)
    X_rus= pd.DataFrame(X_rus, columns=X_h5.columns)
    y_rus= pd.Series(y_rus, name=y.name)

    print('Removed indexes:', id_rus)
    #plot Random under-sampling using PCA
    if plot==True:
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_rus))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_rus), colors, markers):
            plt.scatter( 
                X_pca.loc[y_rus.sort_index()== l, 0],# pc 1
                X_pca.loc[y_rus.sort_index()== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()

    return rus, id_rus, X_rus, y_rus

def random_over_sample(X, y, label="Random over-sampling", plot=False):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X, y)
    X_ros= pd.DataFrame(X_ros, columns=X.columns)
    y_ros= pd.Series(y, name=y.name)
    print(X_ros.shape[0] - X.shape[0], 'new random picked points')
    if plot== True:
        #plot Random over-sampling using PCA
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_rus))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_rus), colors, markers):
            plt.scatter( 
                X_pca.loc[y_rus.sort_index()== l, 0],# pc 1
                X_pca.loc[y_rus.sort_index()== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()
    return X_ros, y_ros, ros
'''
Under-sampling: Tomek links
Tomek links are pairs of very close instances, but of opposite classes. Removing the instances of the majority class of each pair increases the space between the two classes, facilitating the classification process.
'''

#In the code below, we'll use ratio='majority' to resample the majority class.
def undersample_tomek_link(X,y, label= 'Tomek links under-sampling', plot=False):
    tl = TomekLinks(return_indices=True, ratio='all')
    X_tl, y_tl, id_tl = tl.fit_sample(X, y)
    X_tl= pd.DataFrame(X_tl, columns=X.columns)
    y_tl= pd.Series(y_tl, name=y.name)
    if plot== True:
        #print('Removed indexes:', id_tl)
        # plotting using pca
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_tl))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_tl), colors, markers):
            plt.scatter( 
                X_pca.loc[y_tl== l, 0],# pc 1
                X_pca.loc[y_tl== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()
    return X_tl, y_tl, tl, id_tl

'''
Under-sampling: Cluster Centroids
This technique performs under-sampling by generating centroids based on clustering methods. The data will be previously grouped by similarity, in order to preserve information.
'''
def undersample_cluster_centroid(X,y, label= 'Cluster Centroids under-sampling', plot=False,
                                sampling_strategy='auto', random_state=None, 
                          estimator=None, voting='auto', n_jobs=-1, ratio=None):
    
    '''
    voting:str, optional (default=’auto’)
    Voting strategy to generate the new samples:

    If 'hard', the nearest-neighbors of the centroids found using the clustering algorithm will be used.
    If 'soft', the centroids found by the clustering algorithm will be used.
    '''
    cc = ClusterCentroids(sampling_strategy=sampling_strategy, random_state=random_state, 
                          estimator=estimator, voting=voting, n_jobs=n_jobs, ratio=ratio)
    X_cc, y_cc = cc.fit_sample(X, y)
    X_cc= pd.DataFrame(X_cc, columns=X.columns)
    y_cc= pd.Series(y_cc, name=y.name)
    if plot== True:
        # plotting using pca
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_cc))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_cc), colors, markers):
            plt.scatter( 
                X_pca.loc[y_cc.sort_index()== l, 0],# pc 1
                X_pca.loc[y_cc.sort_index()== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()
    return X_cc, y_cc, cc
'''
Over-sampling: SMOTE
SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

We'll use ratio='minority' to resample the minority class.'''


def oversample_SMOTE(X,y, label='SMOTE over-sampling', plot= False):

    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
    X_sm= pd.DataFrame(X_sm, columns=X.columns)
    y_sm= pd.Series(y_sm, name=y.name)
    if plot== True:
        # plotting using pca
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_sm))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_sm), colors, markers):
            plt.scatter( 
                X_pca.loc[y_sm.sort_index()== l, 0],# pc 1
                X_pca.loc[y_sm.sort_index()== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()
    return X_sm, y_sm, smote


'''
Over-sampling followed by under-sampling
Now, we will do a combination of over-sampling and under-sampling, using the SMOTE and Tomek links techniques:
'''

def over_under_SMOTETomek(X, y, label='SMOTE + Tomek links', plot= False):
    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(X, y)
    X_smt= pd.DataFrame(X_smt, columns=X.columns)
    y_smt= pd.Series(y_smt, name=y.name)    
    if plot== True:
        # plotting using pca
        pca = PCA(n_components=2)
        X_pca = pd.DataFrame(pca.fit_transform(X_smt))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y_smt), colors, markers):
            plt.scatter( 
                X_pca.loc[y_smt.sort_index()== l, 0],# pc 1
                X_pca.loc[y_smt.sort_index()== l, 1], # pc 2
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()
    return X_smt, y_smt, smt

def learning_curve(X_train, X_test,y_train, y_test, model=sklearn.svm.SVC(), observations=[50,75,100,125,150]):
    recalls = []
    f1s=[]
    precs=[]
    accs=[]
    
    for n in observations:
        smt = SMOTETomek(ratio='auto')
        smt.fit(X_train, y_train)
        X_resampled, y_resampled= smt.fit_sample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
        y_pred= model.predict(X_test)
        
        f1=f1_score(y_pred= y_pred, y_true=y_test, average='macro')
        acc =accuracy_score(y_pred= y_pred, y_true=y_test)
        prec=precision_score(y_pred= y_pred, y_true=y_test, average='weighted')
        recall=recall_score(y_pred= y_pred, y_true=y_test)
        
        f1s.append(f1)
        accs.append(acc)
        precs.append(prec)
        recalls.append(recall)
    plt.plot(observations, f1s, linewidth=4, color= 'blue', label= 'f1')
    plt.plot(observations, accs, linewidth=4, color= 'red', label= 'aacuracy')
    plt.plot(observations, precs, linewidth=4, color= 'green', label= 'precision')
    plt.plot(observations, recalls, linewidth=4, color= 'orange', label='recalls')
    plt.legend()
    
    plt.title("RandomUnderSampler Learning Curve", fontsize=16)
    plt.gca().set_xlabel("# of Points per Class", fontsize=14)
    plt.gca().set_ylabel("Training Accuracy", fontsize=14)
    sns.despine()
    return f1s,accs,precs,recalls, smt, model