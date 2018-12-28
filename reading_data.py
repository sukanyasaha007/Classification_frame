## Read Data

    #•Reading different file types – It can read csv, parquet, hdf, json, feather, Data file, Database file, SQL database file, Linux / Unix tarball file archive, txt file in a single function
    #•Reading multiple files into single DF-  it can multiple parquet files and convert to a single dataframe


import pandas as pd
import feather
from pathlib import Path

def read_data(data_in= None, path=None,  table_name=" "):      #table name is optional
    
    '''
    accepts file names with directory path and gives out pandas dataframe
    Parameters
    --------
    data_in: file name. If data has to be collected from one single file data_in needs to have the complete file name inclding path
    path : directory name in case data needs to be collected from multiple parquet files eg. '/datascience/home/ssaha/input/c360_customeradt_lexussegmentation_2014_03_31/'. Not required for single file source.
    
    '''
    try:
        if (path!=None):    #read from mutliple parquet files 
            df=pd.DataFrame(None)
            data_dir = Path(path)
            df = pd.concat((pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*.parquet')))
            
        elif data_in.endswith('.csv'):         #csv
            df=pd.read_csv(data_in)
            return df
        elif data_in.endswith('.h5'):         #h5
            df = pd.read_hdf(data_in)
            return df
        
        elif data_in.endswith('.feather'):         #feather
            df = feather.read_dataframe(data_in)
            return df

        elif data_in.endswith('.parquet'):         #parquet
            df = pd.read_parquet(data_in)
            return df
        
        elif data_in.endswith('.json'):       #json 
            df=pd.read_json(data_in)
            
        elif data_in.endwith('.dat'):        #Data file
            df=pd.read_table(data_in)
            
        elif data_in.endwith('.db'):         #Database file
            conn = sqlite3.connect(data_in)
            df = pd.read_sql_query("select * from table_name", conn)
            
        elif data_in.endwith('.dbf'):        #Database file
            conn = sqlite3.connect(data_in)
            df = pd.read_sql_query("select * from table_name", conn)
            
        elif data_in.endwith('.log'):        #log file
            df=pd.read_csv(data_in)
            
        elif data_in.endwith('.txt'):        #txt file
            df=pd.read_csv(data_in)
            
        elif data_in.endwith('.sql'):        #SQL database file
            conn = mdb.connect('localhost','myname','mypass', data_in);
            df = sql.read_frame('select * from table_name', conn)
            
        elif data_in.endwith('.tar'):        #Linux / Unix tarball file archive
            df = pd.read_csv(data_in, compression='gzip')
            
        return df
    except ValueError:
        print("Oops!  That file system is not supported")
        
    return df

