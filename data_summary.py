#Data summary

#This file contains details of Data Summary
    #• Overall summary-summary details about Count of row, columns, no of numeric variables, no of categorical variables etc for entire data set
    #• Null Count Percentage- Percentage of missing value in each variable for entire data set
    #• Count of 0/1 Percentage-  Percentage of zero in all column and  Percentage of  one in each numerical column
    #• Most Frequent Values- most frequent values in a column
    #• Write summary to excel – Creates an excel of all above summary

import numpy as np 
import os
import pandas as pd 
from directoryPath import mlresult_dir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
plt.figure(figsize=(6,3))
matplotlib.rc('xtick', labelsize=7)
from config_param import project_identifier
import data_prep as prep
import datetime

def summary_function(df,p=0.5, file_name_prefix=" "):
    '''
    Takes a dataframe and returns summary details about Count of row, columns, no of numeric variables, no of categorical variables etc for entire data set
    Parameters
    --------
    df: A dataframe
    p: 
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    '''
    df_11=df
    df=df._get_numeric_data()
    df_names=df.columns

    col_nonmiss= df.apply(lambda x: x.count(), axis=0)
    col_miss = df.shape[0] - col_nonmiss
    col_mean=df[df_names].mean(axis=0)
    col_min=df[df_names].min(axis=0)
    col_max=df[df_names].max(axis=0)
    col_std=df[df_names].std(axis=0)
    col_median=df[df_names].median(axis=0)
    col_quantiles=df.quantile(p)

    total_miss_percent= sum(col_miss)
    total_miss_percent=100.0*(total_miss_percent/(df_11.shape[0]*df_11.shape[1]))
    total_nonmiss_percent=abs(100-total_miss_percent)
    col_miss_max=(col_miss.max()/df_11.shape[0])*100.0
    col_miss_min=(col_miss[col_miss!=0].min()/df_11.shape[0])*100.0
    
    n_numnericCol_per=(len(df_names)/len(df_11.columns))*100.00
    n_non_numericCol_per=100.0-n_numnericCol_per
    '''
    hist_array=[]
    hist_array.append(total_miss_percent)
    hist_array.append(total_nonmiss_percent)
    hist_array.append(col_miss_max)
    hist_array.append(col_miss_min)
    hist_array.append(n_numnericCol_per)
    hist_array.append(n_non_numericCol_per)
    '''
    x_label=["Total \n miss (%)","Total \n nonmiss(%)","Max miss \n for Col(%)","Min miss \n for Col(%)", "n_numneric \n Col (%)", "n_non_ \numeric Col (%)"]
    
    #plt.bar(x_label,hist_array, width=0.3)
    
    #plt.xticks(rotation=7)
    #plt.savefig(mlresult_dir+str(project_identifier) +'_data_summary_hist_'+file_name_prefix)
    df_1=col_quantiles.T
    df_2=pd.concat([col_nonmiss,col_miss,col_mean,col_min,col_max,col_std,col_median],axis=1,keys=['Col Nonmiss','col miss','col mean','col min','col max','col std','col median'])
    df_summary=pd.concat([df_2, df_1], axis=1, join_axes=[df_1.index])
    tfile = open(mlresult_dir + 'data_summary.txt', 'a')
    tfile.write(df_summary.to_string())
    tfile.close()
    return df_summary

#Adding more functions 11/13/2018

# count of row, columns, no of numeric variables, no of categorical variables for entire data set
def get_overall_summary(df):
    '''
    Takes a dataframe and returns Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    Parameter: A dataframe
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    '''
    global overall_summary_df
    overall_summary_df= pd.DataFrame(None)
    overall_summary_df['col_count']= [df.shape[1]]
    overall_summary_df['row_count']= [df.shape[0]]
    overall_summary_df['numeric_features_count']= [df.select_dtypes(exclude='O').shape[1]]
    overall_summary_df['categorical_features_count']= [df.select_dtypes(include='O').shape[1]]
    
    # count the columns which are having only one unique value
    count=0
    for col in df.columns:
        if df[col].nunique()==1:
            count+=1
    overall_summary_df['total_col_with_one_unique_value']=[count]
    
    overall_summary_df.index=['Counts']
    return overall_summary_df

#Count of missing values for each column
def get_missing_value_count(df):
    '''
    Takes a dataframe and returns Count of missing value in each column
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    ''' 
    global missing_value_count_df
    missing_value_count_df= pd.DataFrame(prep.null_value_counts(df))
    return missing_value_count_df
#Count of zero for each column
def get_zero_count_in_numeric_cols(df):
    '''
    Takes a dataframe and returns Count of zero value in each column
    Parameter: A dataframe
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    ''' 
    global zero_count_in_numeric_cols_df
    zero_count_in_numeric_cols_df=pd.DataFrame((df.select_dtypes(include='number').isnull().sum().sort_values(ascending= False)/prep.get_numeric_cols(df).shape[1])*100, columns=['Percentage of zeros'])
    
    return zero_count_in_numeric_cols_df

#Count of one for each column
def get_one_count_in_numeric_cols(df):
    '''
    Takes a dataframe and returns Count of zero value in each numerical column
    Parameter: A dataframe
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    ''' 
    global one_count_in_numeric_cols_df
    one_count_in_numeric_cols_df=pd.DataFrame(None)
    for col in df.columns:
        one_count_in_numeric_cols_df[col]=[((df[col]==1).sum()/df.shape[0])*100]
    #one_count_in_numeric_cols.columns='percentage of ones'
    one_count_in_numeric_cols_df=one_count_in_numeric_cols_df.T
    one_count_in_numeric_cols_df.columns=['Percentage of ones']
    one_count_in_numeric_cols_df.sort_values('Percentage of ones', ascending=False, inplace=True)
    return one_count_in_numeric_cols_df
    
#Most Frequent Value for each column
def get_most_frequent_count(df):
    '''
    Takes a dataframe and returns most frequent values in a column
    Parameter: A dataframe
    Returns: A Datframe containing most frequent values in a categorical column for entire data set
    '''    
    global most_frequent_count_df
    most_frequent_count_df=(df.mode(axis=0).head(1).T)
    most_frequent_count_df.columns=['Mode']
    return most_frequent_count_df
#Create the excel of above data
def write_to_excel(df, path= "../mlresults/"):
    '''
    Takes a dataframe and creates an execl of deatils Count of row, columns, no of numeric variables, no of categorical variables etc for entire data set, Percentage of missing value, Percentage of zero in all column and  Percentage of  one in each numerical column,  most frequent value in a column

    Parameter: A dataframe
    Returns: A Datframe containing Count of row, columns, no of numeric variables, no of categorical variables for entire data set
    ''' 
    file_with_path= os.path.join(path, 'Summary ' + str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + '-' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'xlsx') 

    
    get_overall_summary(df)
    get_missing_value_count(df)
    get_zero_count_in_numeric_cols(df)
    get_one_count_in_numeric_cols(df)
    get_most_frequent_count(df)
    writer=pd.ExcelWriter(file_with_path)
    
    
    overall_summary_df.to_excel(excel_writer=writer, sheet_name='Overall_Summary')
    #df.describe().to_excel(excel_writer=writer, sheet_name='Overall_Summary', startrow= (overall_summary_df.shape[0] +2))

    result=pd.concat((missing_value_count_df,most_frequent_count_df,one_count_in_numeric_cols_df, df.describe().T), sort=True)
    result.to_excel(excel_writer=writer, sheet_name='Overall_Summary', startrow= (overall_summary_df.shape[0] + 4))
    writer.save()