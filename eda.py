##  EDA
    #• Histogram-  Plots histogram of a numerical column, normalizes before plotting if parameter normalize if true.
    #• Scatter plots- Scatter plot with respect to x and y
    #• Missing Value Plot- Histogram of missing percentage of features
    #• Plot unique- Histogram of percentage of unique values
    #• Plot bar categorical – Bar plot for categorical variable
    #• Plot correlation- Heatmap of the features with correlations above the correlated threshold in the data.
    #• Plot summary - plots summary details about Count of row, columns, no of numeric variables, no of categorical variables, number of missing value and unique value for entire data set
    #• Plot variable importance- Plots features based on their ranks per some algorithm

import numpy as np
import datetime
import pandas as pd
import sklearn 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from scipy import stats
from scipy.stats import norm, skew
import data_prep as prep
from config_param import project_identifier
from directoryPath import mlresult_dir

# This part was present in data_summary.py
def plot_summary(df,p=0.5, file_name_prefix=" "):
    '''
    Takes a dataframe and plots summary details about Count of row, columns, no of numeric variables, no of categorical variables etc for entire data set
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
    
    hist_array=[]
    hist_array.append(total_miss_percent)
    hist_array.append(total_nonmiss_percent)
    hist_array.append(col_miss_max)
    hist_array.append(col_miss_min)
    hist_array.append(n_numnericCol_per)
    hist_array.append(n_non_numericCol_per)
    
    x_label=["Total \n miss (%)","Total \n nonmiss(%)","Max miss \n for Col(%)","Min miss \n for Col(%)", "n_numneric \n Col (%)", "n_non_ \numeric Col (%)"]
    
    plt.bar(x_label,hist_array, width=0.3)
    
    #plt.xticks(rotation=7)
    plt.savefig(mlresult_dir+str(project_identifier) +'_data_summary_hist_' + str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')


def reset_plot():
    plt.rcParams = plt.rcParamsDefault
def plot_missing(df):
    """
    Histogram of missing percentage in each feature
    Parameter:
    df: dataframe 
    """
    missing_stats= prep.null_value_counts(df)
    reset_plot()
    missing_stats.plot.hist(color = 'blue', edgecolor = 'k', figsize = (5, 3), fontsize = 12)
    plt.ylabel('Frequency', size = 10)
    plt.xlabel('Missing Percentage', size = 10); plt.title('Missing Value Histogram', size = 15);
    plt.savefig("../mlresults/"+str(project_identifier)+ 'Missing_Value_Plot '+ str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')
def plot_unique(df):
    """
    Histogram of percentage of unique values 
    Parameter:
    df: dataframe 
    """
    reset_plot()
    unique_counts = (df.nunique()/df.shape[0])*100
    unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_stats.plot.hist(edgecolor = 'k', figsize = (6,4), fontsize = 12)
    plt.ylabel('Frequency', size = 14)
    plt.xlabel('Unique Values Count', size = 14); plt.title('Unique Values Histogram', size = 16);
    plt.savefig("../mlresults/"+str(project_identifier)+ '_Unique_value_count_Plot_'+ str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')
    
def plot_correlation(df, correlation_threshold=.5):
    """
    Heatmap of the features with correlations above the correlated threshold in the data.

    Notes
    --------
        - Not all of the plotted correlations are above the threshold because this plots
        all the variables that have been idenfitied as having even one correlation above the threshold
        - The features on the x-axis are those that will be removed. The features on the y-axis
        are the correlated feature with those on the x-axis
    Parameters
    --------
    df: Dataframe
    correlation_threshold: correlation threshold above which features need to be removed

    """

    
    # Identify the correlations that were above the threshold
    record_correlation=prep.remove_col_with_corr(df, correlation_threshold)
    corr_matrix=df.corr()
    corr_matrix_plot = corr_matrix.loc[list(set(record_correlation['corr_feature'])), 
                                            list(set(record_correlation['drop_feature']))]

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                linewidths=.25, cbar_kws={"shrink": 0.6})

    ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
    ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));

    ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
    ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));

    plt.xlabel('Features to Remove', size = 8); plt.ylabel('Correlated Feature', size = 8)
    plt.title("Correlations Above Threshold", size = 14)
    
    plt.savefig("../mlresults/"+str(project_identifier)+ '_Correlation_Plot_'+ str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')
    
#Variable importance

# fit model no training data
def plot_feature_ranking(X,y, estimator='lin'):
    '''
    plots feature rankings for all columns
    parameter
    -------
    X: Data frame of Indepenedent columns
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
    # Create the RFE object and rank each feature
    
    lasso=sklearn.linear_model.Lasso()
    ridge=sklearn.linear_model.Ridge()
    svm=sklearn.svm.SVR()
    dt=sklearn.tree.DecisionTreeRegressor()
    rf=sklearn.ensemble.RandomForestRegressor()
    boost=sklearn.ensemble.AdaBoostRegressor()
    lin=sklearn.linear_model.LinearRegression()
    
    estimator_dict= {'lin': lin, 'ridge': ridge, 'lasso': lasso, 'svm': svm, 'dt': dt, 'rf': rf, 'boost':boost} 
    
    rfe = RFE(estimator_dict[estimator], n_features_to_select=100)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    #rank_df_lasso=pd.DataFrame(ranking, index=X_encoded.columns)
    #print(df.tail())
    
    #Plot pixel ranking
    plt.hist(ranking)
    plt.title("Ranking of independent with RFE")
    plt.xlabel('Rank of Features')
    plt.ylabel('Number of Features')

    #feature_importance= pd.Series(xgb.feature_importances_, X.columns)
    plt.savefig("../mlresults/"+str(project_identifier)+ '_feature_ranking_'+ str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')



def plot_histogram(x, title="Title", xlabel='X axis', ylabel='Y axis', normalize= True):
    '''
    parameter:
    X: numerical column
    normalize: whether to normalize x or not
    Title
    X label
    Y label
       
    '''
    if (normalize== True):
        sns.distplot(x , fit=norm);
    # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(x)
        print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        #Now add lagend to the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                    loc='best')
        
    else:
        sns.distplot(x);
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title)

    #Get also the QQ-plot
    #fig = plt.figure()
    #res = stats.probplot(x, plot=plt)
def plot_scatter(x,y, title="Title", xlabel='X axis', ylabel='Y axis'):
    '''
    parameter:
    X: numerical column
    Title
    X label
    Y label
       
    '''
    plt.scatter(x, y)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title)
    plt.show()

def plot_bargraph(x, title="Title", xlabel='X axis', ylabel='Y axis'):
    '''
    parameter:
    X: categorical column
    Title
    X label
    Y label
       
    '''
    count  = x.value_counts()
    
    plt.figure(figsize=(10,5))
    sns.barplot(count.index, count.values, alpha=0.8)
    plt.title(title)
    plt.ylabel(xlabel, fontsize=12)
    plt.xlabel(ylabel, fontsize=12)
    plt.title(title)
    plt.show()

