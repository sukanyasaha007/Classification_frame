# Deeplearning using h2o
# Super learner using rf and gbm
import sys
sys.path.append('/datascience/home/ssaha/')
import h2o
import time
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.regression import H2ORegressionModel
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
parent_dir         = "/datascience/home/ssaha/"
project_identifier = "c360_customeradt_in_market_lexus"
parent_dir_project = os.path.join(parent_dir,'RMModel/') 
mlresult_dir       = os.path.join(parent_dir_project,'mlresults/')
mlobjects          =os.path.join(parent_dir_project,'mlobjects/')

data_dir           = os.path.join(parent_dir_project,'data/')
input_dir          = os.path.join(parent_dir_project,'input/')
import pandas as pd
h2o.init()
# Deeplearning using h2o
def h2o_deeplearning(train,test, target, model_id, 
                                   epochs=50, 
                                   hidden=[10,10], 
                                   stopping_rounds=0,  #disable early stopping
                                   seed=1,
                                   keep_cross_validation_predictions= True
                                   #balance_classes = True
                                  ):
    """
    runs ensembeled models and returns model object
    Parameter: 
    train, test- h2o dataframe of training and test set. 
    target- name of the dependent column inside the train and test set
    params: It should depend on the model you are passing
        You can also pass and of params such as 
        params = {'learn_rate': [i * 0.01 for i in range(1, 11)],
                'max_depth': [i for i in range(2, 11)],
                'sample_rate': [i * 0.1 for i in range(5, 11)],
                'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
    grid_model: Model on which we on to use the grid search
    Returns:
    
    returns a grid model, best model object and the prediction on test data of the best model
    """

    
    start_time          = time.time()
    # Identify predictors and response
    x=train.columns#.tolist()
       
    y = target
    x.remove(y) 
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    
    '''train = h2o.H2OFrame(train)
    test = h2o.H2OFrame(test )'''

       

    # Train and validate a cartesian  the model
    dl_fit = H2ODeepLearningEstimator(model_id=model_id, 
                                   epochs=epochs, 
                                   hidden=hidden, 
                                   stopping_rounds=stopping_rounds, 
                                   seed=seed,
                                   keep_cross_validation_predictions= keep_cross_validation_predictions
                                   #balance_classes = True
                                  )
    dl_fit.train(x=x, y=y, training_frame=train)

    

    # Now let's evaluate the model performance on a test set
    
    performance = dl_fit.model_performance(test)
    pred = dl_fit.predict(test)
    
    print("Test F1:  {0}".format(performance.F1() ))
    print("Test accuracy:  {0}".format(performance.accuracy() ))
    print("Test precision:  {0}".format(performance.precision() ))
    print("Test recall:  {0}".format(performance.recall([0.01, 0.5, 0.99]) ))
    #print("Test confusion_matrix:  {0}".format(performance.confusion_matrix([ "precision", "accuracy", "f1"]) ))
    
    time_end=time.time() - start_time
    #saving logs
    f = open(mlresult_dir +str(project_identifier) +"_log_H2O_model.csv","a")

    f.write("\n Time taken to execute the H2o model is "+
        str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now()) +"\n" +
        ' Accuracy Score'+ str(performance.accuracy()) +"\n"+'F1 Score ' + str(performance.F1())+ "\n"+'precision score ' + 
        str(performance.precision())+"\n"+ 'recall score ' + str( performance.recall([0.01, 0.5, 0.99])) +"\n" )
    
    f.close()
    print("\n Time taken to execute the H2O deep learning model is "+ 
            str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now())+ "\n"             
         )
    
    return dl_fit, pred
# Super learner using rf and gbm
def super_learner(train,test, target, model_id,
                  nfolds = 5,
                 ntrees=10,
                   max_depth=100,
                    min_rows=1000,
                     learn_rate=0.2,
                       fold_assignment="Modulo",
                         keep_cross_validation_predictions=True,
                          seed=1):
    """
    runs ensembeled models and returns model object
    Parameter: 
    train, test- h2o dataframe of training and test set.
    target- name of the dependent column inside the train and test set
    Returns:
    
    returns a gradient boost, random forest and ensemble model object and the prediction on test data
    """
    # Import a sample binary outcome train/test set into H2O
    start_time          = time.time()
    #train = h2o.H2OFrame(train)
    #test = h2o.H2OFrame(test )

    # Identify predictors and response
    x = train.columns
    y = target
    x.remove(y)

    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    # Number of CV folds (to generate level-one data for stacking)
    

    # There are a few ways to assemble a list of models to stack together:
    # 1. Train individual models and put them in a list
    # 2. Train a grid of models
    # 3. Train several grids of models
    # Note: All base models must have the same cross-validation folds and
    # the cross-validated predicted values must be kept.


    # Generate a 2-model ensemble (GBM + RF)

    # Train and cross-validate a GBM
    my_gbm = H2OGradientBoostingEstimator(#distribution="bernoulli",
                                          ntrees=ntrees,
                                          max_depth=max_depth,
                                          min_rows=min_rows,
                                          learn_rate=learn_rate,
                                          nfolds=nfolds,
                                          fold_assignment=fold_assignment,
                                          keep_cross_validation_predictions=keep_cross_validation_predictions,
                                          seed=seed)
    my_gbm.train(x=x, y=y, training_frame=train)


    # Train and cross-validate a RF
    my_rf = H2ORandomForestEstimator(ntrees=ntrees,
                                     nfolds=nfolds,
                                     fold_assignment=fold_assignment,
                                     keep_cross_validation_predictions=keep_cross_validation_predictions,
                                     seed=seed)
    my_rf.train(x=x, y=y, training_frame=train)


    # Train a stacked ensemble using the GBM and GLM above
    ensemble = H2OStackedEnsembleEstimator(model_id=model_id,
                                           base_models=[my_gbm, my_rf])
    ensemble.train(x=x, y=y, training_frame=train)

    # Eval ensemble performance on the test data
    per_stack_test = ensemble.model_performance(test)

    # Compare to base learner performance on the test set
    per_gbm_test = my_gbm.model_performance(test)
    per_rf_test = my_rf.model_performance(test)
    baselearner_best_recall_test = max(per_gbm_test.recall([0.01, 0.5, 0.99]), per_rf_test.recall([0.01, 0.5, 0.99]))
    stack_recall_test = per_stack_test.recall([0.01, 0.5, 0.99])
    print("Best Base-learner Test Recall:  {0}".format(baselearner_best_recall_test))
    print("Ensemble Test Recall:  {0}".format(stack_recall_test))

    # Generate predictions on a test set 
    pred = ensemble.predict(test)
    time_end=time.time() - start_time
    f = open(mlresult_dir +str(project_identifier) +"_log_H2O_model.csv","a")
    
    f.write("\n Time taken to execute the H2O Supre Learner model is "+ 
            str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now())+
            "Best Base-learner Test recall:" + str(baselearner_best_recall_test)+
            "Ensemble Test recall:"+ str(stack_recall_test)
           )
    f.close()
 
    print("\n Time taken to execute the H2O Supre Learner model is "+ 
            str(time_end) + "\n" +"Dated on"+ str(datetime.datetime.now())+
            "Best Base-learner Test recall:" + str(baselearner_best_recall_test)+
            "Ensemble Test recall:"+ str(stack_recall_test)
           )
        
    print("Test F1:  {0}".format(per_stack_test.F1() ))
    print("Test accuracy:  {0}".format(per_stack_test.accuracy() ))
    print("Test precision:  {0}".format(per_stack_test.precision() ))
    print("Test recall:  {0}".format(per_stack_test.recall([0.01, 0.5, 0.99]) ))

    
    return my_gbm, my_rf, ensemble, pred