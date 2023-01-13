import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from env import get_db_url

from sklearn.model_selection import train_test_split
import sklearn.preprocessing


def get_data(sql_db, query):
    '''
        Accepts 2 arguments of string type:
        1: SQL database name
        2: SQL query
        
        Checks if .csv already exists before
        connecting with SQL database again
        
        Saves a .csv file of DataFrame
        
        Returns DataFrame
    '''
    
    # variable to hold filename created from 
    # input argument of SQL database name
    path = f'{sql_db}.csv'
    
    # Holds boolean result of check for
    # .csv existing; uses OS module
    file_exists = os.path.exists(path)
    
    # Uses boolean value variable to
    # check whether to create a new
    # SQL connection or load .csv
    #
    # Finished off by returning DataFrame
    if file_exists:
        df = pd.read_csv(path)
        
        print('Reading CSV')
        return df

    else:
        url = get_db_url(sql_db)
        df = pd.read_sql(query, url)
        df.to_csv(f'{sql_db}.csv')
        
        print('Downloading SQL DB')
        return df

def clean_data(df):
    '''
        Accepts DataFrame from get_data() function in wrangle.py
            &
        Returns a cleaned DataFrame
    '''

    # Drop Nulls
    df = df.dropna()

    # Temporarily converts 'fips' column to interger to remove
    # trailing zeroes from current float type
    df['fips'] = df['fips'].apply(int)

    # Converts 'fips' column to string to target data type
    df['fips'] = df['fips'].apply(str)

    # Adds leading 'zero' character to 'fips' column, which
    # is now a string data type
    df['fips'] = '0' + df['fips']

    # Converts 'yearbuilt' column to interger
    df['yearbuilt'] = df['yearbuilt'].apply(int)

    # Method of removing last 2 strings in all columns
    #df['fips']=df['fips'].str[:-2]

    return df

def split_data(df):    
    '''
        Accepts a DataFrame 
    
        Splits DataFrame into a train, validate, and test set
        and it will return three values:
        
        train, val, test (in this order) -- all pandas Dataframes
    '''
    
    train, test = train_test_split(df, 
                               train_size = 0.8,
                               random_state=42)
    train, val = train_test_split(train,
                             train_size = 0.8,
                             random_state=42)
    return train, val, test

def wrangle_zillow():
    '''
        Main function in `wrangle.py`
        When run, wrangle_zillow will utilize
        get_db_url(), get_data(), and clean_data()
        
        to acquire & prepare DataFrame
        
        returns a DataFrame
    '''
    
    sql_db = "zillow"
    query = '''
            SELECT
                bedroomcnt,
                bathroomcnt,
                calculatedfinishedsquarefeet,
                taxvaluedollarcnt,
                yearbuilt,
                taxamount,
                fips,
                propertylandusedesc
            FROM 
                properties_2017
                JOIN 
                    propertylandusetype USING(propertylandusetypeid)
            WHERE 
                propertylandusedesc = 'Single Family Residential';
            '''
    df = get_data(sql_db,query)
    df = clean_data(df)
    
    return df

def MinMaxScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Min Max Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def StandardScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Standard Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def RobustScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Robust Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def compare_scalers(df):
    '''
        Accepts a DataFrame
        
        Is used to visualize 3 scaler outputs
        and compare to original DataFrame
    '''
    mm_scaled = MinMaxScaler(df)
    ss_scaled = StandardScaler(df)
    rs_scaled = RobustScaler(df)

    font = {'family': 'Georgia',
            'color':  '#525252',
            'weight': 'bold',
            'size': 25,
            }
    # ====================================================================

    # Assigning 'fig', 'ax' variables.
    fig, ax = plt.subplots(2, 2,figsize=(25,25))

    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (0, 8000)

    # Setting the values for all axes.
    #plt.setp(ax, xlim=custom_xlim)
    
    # ====================================================================
    
    # Original Data
    ax[0][0].hist(df, color="#525252",ec='white',bins=10000)
    ax[0][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_title("Original",color='#525252', fontdict=font)
    ax[0][0].set_xlim([0, 8000])
    
    
    # MinMax Scaled
    ax[0][1].hist(mm_scaled, color="#525252",ec='white',bins=10000)
    ax[0][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_title("MinMax Scaled",color='#525252', fontdict=font)
    ax[0][1].set_xlim([0, .005])
    
    # Standard Scaled
    ax[1][0].hist(ss_scaled, color="#525252",ec='white',bins=10000)
    ax[1][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_title("Standard Scaled",color='#525252', fontdict=font)
    ax[1][0].set_xlim([-1.5, 3])

    # Robust Scaled
    ax[1][1].hist(rs_scaled, color="#525252",ec='white',bins=10000)
    ax[1][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_title("Robust Scaled",color='#525252', fontdict=font)
    ax[1][1].set_xlim([-2, 5])
