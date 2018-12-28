
## Feature Engineering
    #• Log Transformations – Log Transformation of a column
    #• Reciprocal Transformations- Reciprocal Transformation of a column
    #• Square Transformations- Square Transformation of a column
    #• Find skewness- gives details of skewness with respect to features
    #• Transform Skewed Data- Transforms the skewed variables per Transformation parameter 

from scipy import stats
import time
import datetime
from scipy.stats import norm, skew, boxcox
from scipy.special import boxcox1p
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def trans_boxcox(x, trans=0):
    '''
    parameter: 
    X: a vector or dataframe column
    trans
     = -1. is a reciprocal transform.
     = -0.5 is a reciprocal square root transform.
     = 0.0 is a log transform.
     = 0.5 is a square root transform.
     = 1.0 is no transform.
    '''
    start_time          = time.time()
    x_trans = boxcox(data, trans)
    # histogram
    #plt.hist(data)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_feature_transformation.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the boxcox function is "+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return x_trans
def binning():
    pass

def find_skewness(X):
    '''
    parameter: 
    X : feature matrix
    
    '''
    start_time          = time.time()
    numeric_feats = X.select_dtypes(exclude='O').columns

    # Check the skew of all numerical features
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_feature_transformation.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the find skewness function is "+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" )
    return skewness

def trans_skewed(X, thresh=0.75, lmbda = 0):
    '''
    Power or log transformation
    y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
    log(1+x)                    if lmbda == 0
    parameter: 
    X : feature matrix
    thresh: threshold for skewness
    '''
    start_time          = time.time()
    skewness= find_skewness(X)
    skewness = skewness[abs(skewness) > thresh]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    skewed_features = skewness.index
    
    for feat in skewed_features:
        #all_data[feat] += 1
        X[feat] = boxcox1p(X[feat], lmbda)
    time_end            =time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_feature_transformation.csv","a")
    f.write(sttime + '\n')
    f.write("\n Time taken to execute the 'transform skewed fetures' function is "+ str(time_end) + "\n" )
    f.close()
    print("\n Time taken to execute the function is "+ str(time_end) + "\n" + "Feature having skewness more than threshold value are" + str(skewed_features)+ '\n')
    return X