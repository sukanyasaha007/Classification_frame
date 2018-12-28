# It contains functions to keep log of data summary and model output


import os
import sys
import time
import datetime
from directoryPath import parent_dir, data_dir, parent_dir_project, mlresult_dir
from config_param import project_identifier 
import data_summary
import data_split

#print (mlresult_dir+ "log_result.txt")
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')
'''
def log_record_result( alogorithm_name, time_taken, r2score_train, r2score_test, adjr2score_train,adjr2score_test, mrs_train=None, mrs_test=None, best_param=None):
     
    f = open(mlresult_dir + str(project_identifier)+ "_log_algo_result.txt","a")
    f.write(sttime + '\n')
    f.write(str(alogorithm_name)+"\t"+"total time"+"\t"+ str(time_taken) +"\t" +"score:"+ "\t\t" +"R2 train score =" + "\t" +str(r2score_train)+"\t"+"R2 test score =" +"\t" +str(r2score_test)+"\t"+"adjusted R2 train score ="  +"\t" + str(adjr2score_train)+"\t" +"adjusted R2 test score ="+ "\t" +str(adjr2score_test) +"\t" +"mrs train score ="+"\t"+str(mrs_train)+"\t"+"mrs test score ="+"\t"+str(mrs_test)+"\t"+"best param ="+"\t"+str(best_param)+"\t" +"dated at "+ "\t" + str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year)+"\n")
    f.close()
'''        

def log_data_summary( df ):
    with open(os.path.join(mlresult_dir + str(project_identifier)+ "_log_summary.txt"), 'a' ) as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow("Summary")
        data_summary.get_overall_summary(df).to_csv(f, sep=':', mode='a', line_terminator='\n \n')
        writer.writerow("missing_value_count")
        data_summary.get_missing_value_count(df).to_csv(f, sep=':', mode='a', line_terminator='\n \n')
        writer.writerow("zero_count_in_numeric_cols")
        data_summary.get_zero_count_in_numeric_cols(df).to_csv(f, sep=':', mode='a', line_terminator='\n \n')
        writer.writerow("one_count_in_numeric_cols")
        data_summary.get_one_count_in_numeric_cols(df).to_csv(f, sep=':', mode='a', line_terminator='\n \n')
        writer.writerow("most_frequent_count")
        data_summary.get_most_frequent_count(df).to_csv(f, sep=':', mode='a', line_terminator='\n \n')
        
        
