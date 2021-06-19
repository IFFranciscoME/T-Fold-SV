
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- T-Fold-SV is Time Series Folds for Sequential Validation, the go to alternative for K-Fold-CV       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Description: Python Implementation of the T-Fold Sequential Validation Method                       -- #
# -- synthetics.py: Synthetic data generation for exploration and testing                                -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/T-Fold-SV                                              -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ------------------------------------------------------------------------------- RANDOM WALK WITH DRIFT -- # 
# ------------------------------------------------------------------------------- ---------------------- -- # 

# Single value
np.random.seed(123) 
mu, sigma = 0.1, 0.1 # mean and standard deviation
s = np.cumsum(np.random.normal(mu, sigma, 100))
