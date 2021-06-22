
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
from continuous_futures import futures_data

# ----- T-Fold V0.1 Process ----- # 
# ------------------------------- #

# -- 0. Load Global Dataset
global_data = futures_data.copy()

# -- 1. Folds Formation
folds_data = dt.folds_formation(global_data=global_data, fold_size='year')

# -- 2. Target and Features Engineering

# Target Variable: Time-based labeling

# -- Tests for target variable formation and KLD metric for continuous variables
target_2010 = fn.ohlc_labeling(ohlc_data=folds_data['y_2010'], p_label='co')
target_2011 = fn.ohlc_labeling(ohlc_data=folds_data['y_2011'], p_label='co')
kld_metric = fn.kld(p_data=target_2010, q_data=target_2011, prob_dist='gamma')

# Feature Engineering: In-Fold + Heuristic

# -- 3. Information Assesment
# -- 4. Learning Assesment
# -- 5. Generalization Assesment

# ------------------------------- #
