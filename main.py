
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- T-Fold-SV is Time Series Folds for Sequential Validation, the go to alternative for K-Fold-CV       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Description: Python Implementation of the T-Fold Sequential Validation Method                       -- #
# -- main.py: python script with main method                                                             -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/T-Fold-SV                                              -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import pandas as pd
import numpy as np

# -- Load other scripts
import data as dt
import functions as fn

# -- Read Future Continuous Prices -- #

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

# -- 0. Load Global Dataset (Downsample price data)
global_data = dt.resample_data(target_data=data_futures.copy(), target_freq='8H',
                               auto_names=True, col_names=None)

"""
Pre-Building

- an OHLCV timestamp-indexed dataset.
- Specify the fold_size parameter.
- Specification for labeling the target variable.
- Specification for feature engineering process.
"""

# ----- T-Fold-SV ----- # 
# --------------------- #
"""

- Information Leakage
- - Criteria specification (embargo and purge)

- Information Tensor
- - PDF with target variable in every Fold
- - KLD with all combinations of Folds
- - Information Threshold

- Sparsity assesment
- - Classify Information tensor as: Sparse, Weakly Sparse, Non-Sparse

- Learning Process
- - Define type of learning: Sparse, Weakly Sparse, Non-Sparse
- - Model definition (parameters, hyperparameters)
- - Conduct Optimization and Learning processes

- Performance assesment
- - Define performance metric (Context dependant)
- - Out-of-Sample generalization assesment
- - Out-of-Distribution generalization assesment

Type 2 requires:

- A timestamp-indexed dataset
- At least one target variable, and one or more Features.
- Specify the fold_size parameter.
- Specify Information Leakage Prevention Criteria.
"""

# ---------------------------------------------------------------------------------- T-FOLD-SV : TYPE 1 -- #  
# ---------------------------------------------------------------------------------- ------------------ -- #  

# -- 1. Folds Formation
folds_data = dt.folds_formation(global_data=global_data, fold_size='year')

# -- 2. Target and Features Engineering

# Target Variable: Time-based labeling
Targets = {}
for label in list(folds_data.keys()):
    Targets[label] = fn.ohlc_labeling(ohlc_data=folds_data[label], p_label='co')

# Feature Engineering
Features = {}
for label in list(folds_data.keys()):
    Features[label] = fn.ohlc_features(ohlc_data=folds_data[label], p_label='co')

# -- Tests for target variable formation and KLD metric for continuous variables
target_2010 = fn.ohlc_labeling(ohlc_data=folds_data['y_2010'], p_label='co')
target_2011 = fn.ohlc_labeling(ohlc_data=folds_data['y_2011'], p_label='co')
kld_metric = fn.kld(p_data=target_2010, q_data=target_2011, prob_dist='gamma')

# Feature Engineering: In-Fold + Heuristic

# -- 3. Information Assesment
# -- 4. Learning Assesment
# -- 5. Generalization Assesment

# ------------------------------- #
