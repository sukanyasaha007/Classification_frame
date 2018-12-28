
## Model Interpretation
    #Lime has been used to compare the effect of features on some prediction based on an algorithm

import os
import sys
import time
import datetime
import lime
import lime.lime_tabular
project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def lime_explainer(X_train, y_train, X_test, y_test, model_predictor,df_row=2, alogorithm_name=" ", categorical_features_list= None):
    #df_Xtrain is the training features excluding target
    #df_row is observation to be explained. It is 1D array
    #model_predictor is the model dependent regresser

    explainer  = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values, training_labels= y_train,
                                                    feature_names=X_train.columns.tolist(),
                                                    categorical_features=categorical_features_list, 
                                                    verbose=True, mode='regression')
    
    exp = explainer.explain_instance(X_test.iloc[df_row,:], model_predictor.predict)
    exp.show_in_notebook(show_table=True)
    exp_result = exp.as_list()
    

    f = open(mlresult_dir + str(project_identifier) +"_log_lime.txt","a")
    f.write(sttime + '\n')
    f.write(str(alogorithm_name)+"\t"+str(exp_result)+ "\n")
    f.close()
    print("\n" + "Lime Output " + str(exp_result))

