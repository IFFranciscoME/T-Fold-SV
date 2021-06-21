
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- T-Fold-SV is Time Series Folds for Sequential Validation, the go to alternative for K-Fold-CV       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Description: Python Implementation of the T-Fold Sequential Validation Method                       -- #
# -- data.py: read and transform real future contracts data                                              -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/T-Fold-SV                                              -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load packages for this script
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------- RESAMPLE OHLC PRICES -- #
# ------------------------------------------------------------------------------------------------------ -- #

def resample_data(target_data, target_freq='D', auto_names=True, col_names=None):
    """
    Downsample price data from high frequency, e.g. minute prices, to low frequency, e.g. dayly prices. 

    Parameters
    ----------
    
    target_data: DataFrame

        The OHLC price data to be resampled, currently, only downsample operations are supported, e.g from minute to daily prices, if an upsample operation is detected, e.g. from daily to minute prices, a ValueError is raised. It could be a string to specify the route of the file to be read, or, an existing pd.DataFrame object.

        DataFrame with a column that can be converted to be a timestamp by the .to_datetime() method, if not present a ValueError will be raised. Also other columns are needed, the open, high, low, close prices, and a volume column is optional, all values, except timestamp, will be coherced to numeric. The column names for the output object will stay the same as input object, unless auto_names is set to False and a list of names is provided through col_names parameter.

    target_freq: str

        The target frequency to resample the prices, the options are:
            'H1': to an hour interval, 'H8': to 8 hours interval, 'D': to a day interval.        

    auto_names: bool

        True (Default): Use provided generic columns names: ['timestamp', 'open', 'high', 'low', 'close']
        False: Use the ones in the input object
    
    col_names: list

        None: (Default) 
        list: If auto_names is set to False, a list of every column is expected.

    Returns
    -------    

    r_grouped_data: DataFrame
        
        DataFrame with the new resampled data

    Example
    -------

    >> data = pd.DataFrame({'timestamp': ['2020-01-01 10:00:00', '2020-01-01 10:00:00'],
                            'open': [1.25, 1.25], 'high': [1.35, 1.35],
                            'low': [1.15, 1.15], 'close': [1.30, 1.30], 'vol': [123, 123]})
    >> auto_names = True
    >> col_names = None
    >> target_freq = 'D'

    """
      
    # -- Read input object
    data = target_data.copy()
    
    # -- Column names and types cohercion
    
    # OHLC columns
    if len(list(data.columns)) < 5 or len(list(data.columns)) > 6:
        raise IndexError('There are less than 5 or more than 6 columns in the DataFrame')
    
    # Open, High, Low, Close Columns
    elif len(list(data.columns)) == 5:
        if auto_names:
            data.columns = ['timestamp', 'open', 'high', 'low', 'close']
    
    # Open, High, Low, Close, Volume Columns
    elif len(list(data.columns)) == 6:
        if auto_names:
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Timestamp data conversion
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.index = data['timestamp']
    data.drop('timestamp', inplace=True, axis=1)

    # -- Resampling process
    conversion = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    r_grouped_data = data.resample(target_freq).agg(conversion)

    # Eliminar los NAs originados por que ese minuto
    r_grouped_data = r_grouped_data.dropna()

    return r_grouped_data

# ------------------------------------------------------------------------------------ T-FOLDS FORMATION -- #
# --------------------------------------------------------------------------------------------------------- #

def folds_formation(global_data, fold_size):
    """
    Function to separate in T-Folds the data, the functions guarantees not having filtrations
    
    Parameters
    ----------
    global_data : pd.DataFrame
        DataFrame with a global dataset
    
    fold_size : str
        'month': every T-Fold will be of one month of historical data
        'quarter': every T-Fold will be of three months (quarter) of historical data
        'year': every T-Fold will be of twelve months of historical data
        'bi-year': every T-Fold will be of 2 years of historical data
        '80-20': Hold out method, 80% for training and 20% for valing
    
    Returns
    -------
    n_data: dict
        A dictionary with the data separated by Fold, stored by keynames according to Fold size
    
    """

    # For quarterly separation of the data
    if fold_size == 'quarter':
        # List of quarters in the dataset
        quarters = list(set(time.quarter for time in list(global_data.index)))
        # List of years in the dataset
        years = set(time.year for time in list(global_data.index))
        q_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            q_data.update({'q_' + str('0') + str(i) + '_' + str(y) if i <= 9 else str(i) + '_' + str(y):
                               global_data[(global_data.index.year == y) & (global_data.index.quarter == i)]
                           for i in quarters})
        return q_data

    # For semester separation of the data
    elif fold_size == 'semester':
        # List of years in the dataset
        years = set(time.year for time in list(global_data.index))
        s_data = {}
        # New key for every semester_year
        for y in sorted(list(years)):
            s_data.update({'s_' + str('0') + str(1) + '_' + str(y):
                               global_data[(global_data.index.year == y) & ((global_data.index.quarter == 1) |
                                      (global_data.index.quarter == 2))]})

            s_data.update({'s_' + str('0') + str(2) + '_' + str(y):
                               global_data[(global_data.index.year == y) &
                                      ((global_data.index.quarter == 3) | (global_data.index.quarter == 4))]})

        return s_data

    # For yearly separation of the data
    elif fold_size == 'year':
        # List of years in the dataset
        years = set(time.year for time in list(global_data.index))
        y_data = {}
        # New key for every year
        for y in sorted(list(years)):
            y_data.update({'y_' + str(y):
                               global_data[(global_data.index.year == y)]})
                               
        return y_data

    # For yearly separation of the data
    elif fold_size == '80-20':
        # List of years in the dataset
        years = sorted(list(set(time.year for time in list(global_data.index))))
        # dict to store data
        a_8 = int(len(years)*0.80) 
        a_2 = int(len(years)*0.20)
        # data construction
        data_80_20 = {'h_8': global_data[global_data.index.year.isin(years[0:a_8])],
                      'h_2': global_data[global_data.index.year.isin(years[a_8:a_8+a_2])]}
    
        return data_80_20
