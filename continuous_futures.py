
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

# -- Load other scripts
import data as dt

# ------------------------------------------------------------------------------------ READ FUTURES FILE -- # 

# File name and route
file_name = 'MP_H1_2010_2021'
file_route = 'files/prices/' + file_name + '.txt'

# Read files and prepare format
data_futures = pd.read_csv(file_route, header=None,
                           names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Conversion from Usd per Mxn to Mxn per Usd
data_futures['timestamp'] = pd.to_datetime(data_futures['timestamp'])
data_futures['open'] = round(1/data_futures['open'], 5)
data_futures['high'] = round(1/data_futures['high'], 5)
data_futures['low'] = round(1/data_futures['low'], 5)
data_futures['close'] = round(1/data_futures['close'], 5)

# Swap high and low since the exchange rate was swapped from MXNUSD to USDMXN
low = data_futures['low'].copy()
high = data_futures['high'].copy()
data_futures['high'] = low
data_futures['low'] = high

# Clean memory for these dummy objects
high, low = None, None

# -------------------------------------------------------------------------------------- RESAMPLE PRICES -- #

# Downsample data
futures_data = dt.resample_data(target_data=data_futures.copy(), target_freq='8H',
                                auto_names=True, col_names=None)
