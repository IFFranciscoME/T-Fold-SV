
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

# ------------------------------------------------------------------------------------ READ FUTURES FILE -- # 
# ------------------------------------------------------------------------------------------------------ -- #

# File route
file_route = 'files/prices/MP_H1.txt'

# Read files and prepare format
data_f = pd.read_csv(file_route, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Conversion from Usd per Mxn to Mxn per Usd
data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])
data_f['open'] = round(1/data_f['open'], 5)
data_f['high'] = round(1/data_f['high'], 5)
data_f['low'] = round(1/data_f['low'], 5)
data_f['close'] = round(1/data_f['close'], 5)

# Swap high and low since the exchange rate was swapped from MXNUSD to USDMXN
low = data_f['low'].copy()
high = data_f['high'].copy()
data_f['high'] = low
data_f['low'] = high

# Clean memory for these dummy objects
high, low = None, None
