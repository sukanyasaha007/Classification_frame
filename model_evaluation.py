#cross_val_score, 
#accuracy_score, 
#balanced_accuracy_score, 
#f1_score, precision_score,
#recall_score, 
#Log loss, 
#roc_curve, 
#roc_auc_score,
#confusion_matrix

import sys
import time
import sys
import warnings
import datetime
import gc
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import os

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, log_loss, recall_score, roc_curve, roc_auc_score, confusion_matrix,classification_report 
from sklearn.metrics import  adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import fbeta_score

project_identifier = "c360_customeradt_in_market_lexus"
parent_dir         = "/datascience/home/ssaha/"

parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')

#model_evaluation

def model_evaluation(X_train,y_train, X_test, y_test,y_pred,  model, time_end,  alg_name='Model'):
    # Receive a fitted model
    
    #Find cross_val_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score,recall_score, Log loss, roc_curve, roc_auc_score,confusion_matrix
    #cross_val_score
    print('cross_val_score ' +str(cross_val_score(model, X_train, y_train, cv=5)))
    # calculate the scores
    #accuracy_score
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
    y_pred_proba = model.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_test, y_score=y_pred_proba)

    plt.plot([0,1],[0,1],'k--')
    plt.plot(false_positive_rate,true_positive_rate, label=alg_name)
    plt.xlabel('false_positive_rate')
    plt.ylabel('true_positive_rate')
    plt.title('ROC curve of ' + alg_name)
    plt.show()
    
    #roc_auc_score
    roc= roc_auc_score(y_test,y_pred_proba) * 100
    print('roc_auc_score ' + str(roc))
    
    #confusion_matrix
    print('confusion_matrix ' + str(confusion_matrix(y_test,y_pred)))
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    class_report= classification_report(y_test,y_pred)
    print("classification_report"+ str(class_report))
    f = open(mlresult_dir +str(project_identifier) +"_log_Models.csv","a")
    
    f.write("\n Time taken to execute the "+ alg_name + " is "+
            str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now()) +"\n" +
            ' Accuracy Score'+ str(acc) +"\n"+'f1_score ' + str(f1)+ "\n"+'precision_score ' + 
            str(prec)+"\n"+ 'recall_score ' + str( recall) +"\n"+'roc_auc_score ' + str(roc)+"\n" +
            'classification_report '+ str(class_report))
    f.close()
    print("\n Time taken to execute the "+ alg_name + " is " + str(time_end))
    
    print("\n" +"Dated on"+ str(datetime.datetime.now())+ "\n")
