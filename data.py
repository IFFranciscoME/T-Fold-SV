
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

# -- Load libraries for script
import numpy as np
import pandas as pd
from os import listdir, path, environ
from os.path import isfile, join

# --------------------------------------------------------------- READ AND CONCATENATE FILES FROM FOLDER -- # 

main_path = 'files/prices/'
abspath_f = path.abspath(main_path)
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

# iterative data reading
for file_f in files_f:
    # read files and prepare format
    data_f = pd.read_csv(main_path + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])
    data_f['open'] = round(1/data_f['open'], 5)
    data_f['high'] = round(1/data_f['high'], 5)
    data_f['low'] = round(1/data_f['low'], 5)
    data_f['close'] = round(1/data_f['close'], 5)
    
    # swap high and low since the exchange rate was swapped from MXNUSD to USDMXN
    low = data_f['low'].copy()
    high = data_f['high'].copy()
    data_f['high'] = low
    data_f['low'] = high

    # Clean memory for these dummy objects
    high, low = None, None

    # rename keys to include the year of the data subset (read file)
    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['H8_' + year_f] = data_f
