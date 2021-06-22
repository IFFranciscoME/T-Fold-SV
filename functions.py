
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- T-Fold-SV is Time Series Folds for Sequential Validation, the go to alternative for K-Fold-CV       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Description: Python Implementation of the T-Fold Sequential Validation Method                       -- #
# -- functions.py: Mathematical and Data Processing functions                                            -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/T-Fold-SV                                              -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np
import pandas as pd
import scipy.special as sps

# ---------------------------------------------------------------------------------------- OHLC LABELING -- #
# --------------------------------------------------------------------------------------------------------- #

def ohlc_labeling(ohlc_data, p_label='b_co'):
    """
    This methods offers some options to generate target variables according to the selected labeling process. The input, OHLC prices, is already time-based labelled, nevertheless, a lower granularity labeling process can be conducted. It is recomended though to use this function with higher frequency prices (by the minute or more frequent if possible).

    Normally, a numeric value like co = close - open will be utilized for a regression type of problem, whereas the binary classification problem can be formulated as the sign operation for co, having therefore a 1 if close > open and 0 otherwise. For numerical stability of some cost functions, it is better to have 1 and 0 instead of 1 and -1.

    Parameters
    ----------

    ohlc_data: DataFrame
        With at least 4 numeric columns: 'open', 'high', 'low', 'close'
    
    p_label: str
        An indication of the labelling function, must chose one of the following options:
            'co': close - open
            'b_co': binary version of 'co', i.e. sign[close - open], 1 if close > open, 0 otherwise

    Returns
    -------

        labels according to selected method, currently the results will be:
            - numeric when selecting 'co'
            - numeric and binary (1s and 0s) when selecting 'b_co'

    Example
    -------

    >>> ohlc_data = pd.DataFrame({'timestamp': ['2020-01-01 10:00:00', '2020-01-01 10:00:00'],
                                 'open': [1.25, 1.25], 'high': [1.35, 1.35],
                                 'low': [1.15, 1.15], 'close': [1.30, 1.30], 'vol': [123, 123]})
    >>> label = 'b_co'

    """

    # shallow copy of data
    f_data = ohlc_data.copy()
    # base calculation
    co = f_data['close'] - f_data['open']
    # return continuous variable
    if p_label == 'co':
        return co
    # return discrete-binary variable
    elif p_label == 'b_co':
        return pd.Series([1 if i > 0 else 0 for i in list(co)], index=f_data.index)
    
    # raise error
    else:
        raise ValueError("Accepted values for label are: 'co' or 'b_co' ")

# -------------------------------------------------------------------------- KULLBACK-LEIBLER DIVERGENCE -- #
# --------------------------------------------------------------------------------------------------------- #

def kld(p_data, q_data, prob_dist, pq_shift=True):
    """
    Computes the divergence between two empirical adjusted probability density functions.

    Parameters
    ----------

    p_data: np.array
        Data of the first process

    q_data: np.array
        Data of the first process
    
    prob_dist: str
        Probability distribution - Added: to fit to empirical data
        'gamma': Generalized gamma distribution
    
    pq_shift: bool
        True (Default): Shifts the data in order to have only positive values. This is done by adding, to all values, the absolute of the most negative value.
    
    Returns
    -------
    
    r_kld_gamma: numeric
            Kullback-Leibler Divergence Metric

    References
    ----------

    [1] Kullback, S., & Leibler, R. (1951). On Information and Sufficiency. The Annals of Mathematical  Statistics, 22(1), 79-86. Retrieved June 21, 2021, from http://www.jstor.org/stable/2236703

    Example
    -------

    >>> data_p = np.random.default_rng().gamma(2, 1, 100)
    >>> data_q = np.random.default_rng().gamma(1, 2, 100)
    >>> kld_metric = kld(p_data=data_p, q_data=data_q, prob_dist='gamma')

    """

    # Shift data to have only positive values
    if pq_shift:
        q_data = (q_data + abs(min(q_data)))/max(q_data)
        p_data = (p_data + abs(min(p_data)))/max(p_data)

    # -- with Gamma Distribution -- #
    # ----------------------------- #

    # For continuous variables
    if prob_dist == 'gamma':
        return _kld_gamma(p_data=p_data, q_data=q_data)

    # -- with Binomial Distribution -- #
    # -------------------------------- #

    # For discrete variables (Pending)

    # -- with Other Distribution -- #
    # ----------------------------- #

    else:
        return print('error')

# ----------------------------------------------------------------------- KLD with generalized gamma -- #

def _kld_gamma(p_data, q_data):
    """
    Computes the Kullback-Leibler divergence between two gamma PDFs
    
    Parameters
    ----------

    p_data: np.array
        Data of the first process

    q_data: np.array
        Data of the first process

    Returns
    -------

    r_kld_gamma: numeric
        Kullback-Leibler Divergence Quantity
    
    References
    ----------
    [1] Bauckhage, Christian. (2014). Computing the Kullback-Leibler Divergence between two Generalized Gamma Distributions. arXiv. 1401.6853. 
    
    """
    
    # -------------------------------------------------------------------------- Distribution Parameters -- #

    def _gamma_params(data, method='MoM'):
        """
        Computes the parameters of a gamma probability density function (pdf), according to the selected
        method.

        Parameters
        ----------

        data: np.array
            The data with which will be adjusted the pdf
        
        method: str
            Method to calculate the value of the parameters for the pdf
                'MoM': Method of Moments (Default)

        Returns
        -------

        r_params: dict
            {'alpha': gamma distribution paramerter, 'beta': gamma distribution parameter}
        
        """

        # -- Methods of Moments -- #
        if method == 'MoM':

            # first two moments
            mean = np.mean(data)
            variance = np.var(data)
            # sometimes refered in literature as k
            alpha = mean**2/variance
            # sometimes refered in literature as 1/theta
            beta = mean/variance
            # return the gamma distribution empirically adjusted parameters
            return alpha, beta
        
        # -- For errors or other unsupported methods
        else:
            raise ValueError("Currently, the supported methods are: 'MoM'")

    # alpha_1: Distribution 1: shape parameter, alpha_1 > 0
    # beta_1:  Distribution 1: rate or inverse scale distribution parameter, beta_1 > 0
    alpha_1, beta_1 = _gamma_params(data=p_data)

    # alpha_2: Distribution 2: shape parameter, alpha_2 > 0
    # beta_2:  Distribution 2: rate or inverse scale parameter, beta_2 > 0  
    alpha_2, beta_2 = _gamma_params(data=q_data)

    # Expression with beta instead of theta
    theta_1 = 1/beta_1
    theta_2 = 1/beta_2
    p1, p2 = 1, 1  # Generalized Gamma Distribution with p=1 is a gamma distribution [1]
    
    # Calculations, see [1] for mathematical details.
    a = p1*(theta_2**alpha_2)*sps.gamma(alpha_2/p2)
    b = p2*(theta_1**alpha_1)*sps.gamma(alpha_1/p1)
    c = (((sps.digamma(alpha_1/p1))/p1) + np.log(theta_1))*(alpha_1 - alpha_2)
    
    # Bi-gamma functions
    d = sps.gamma((alpha_1+p2)/p1)
    e = sps.gamma((alpha_1/p1))
    
    # Calculations
    f = (theta_1/theta_2)**(p2)
    g = alpha_1/p1
    
    # General calculation and output
    r_kld = np.log(a/b) + c + (d/e)*f - g  

    # Final Kullback-Leibler Divergence for Empirically Adjusted Gamma PDFs
    return r_kld
