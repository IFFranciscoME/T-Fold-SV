
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
import scipy

# -------------------------------------------------------------------------- KULLBACK-LIEBLER DIVERGENCE -- #
# --------------------------------------------------------------------------------------------------------- #

def kld(p_data, q_data, prob_dist):
    """
    Computes the Divergence between two empirical probability density functions adjusted to the data. 

    Parameters
    ----------

    p_data: np.array
        Data of the first process

    q_data: np.array
        Data of the first process
    
    prob_dist: str
        Probability distribution to fit to empirical data
        'gamma': Generalized gamma distribution

    Returns
    -------
    
    r_kld_gamma: numeric
            Kullback-Liebler Divergence Metric

    References
    ----------

    [1] 

    [2] 

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
                'MLE': Maximum Likelihodd

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
            
            return alpha, beta
        
        # -- Maximum Likelihood -- #
        elif method == 'MLE':

            return 'coming soon'

    # ----------------------------------------------------------------------- KLD with generalized gamma -- #

    def _kld_gamma(alpha_1, beta_1, alpha_2, beta_2):
        """
        Computes the Kullback-Leibler divergence between two gamma PDFs

        Parameters
        ----------
        alpha_1: numeric
            Distribution 1: shape parameter, alpha_1 > 0

        beta_1: numeric
            Distribution 1: rate or inverse scale distribution parameter, beta_1 > 0

        alpha_2: numeric
            Distribution 2: shape parameter, alpha_2 > 0

        beta_2: numeric
            Distribution 2: rate or inverse scale parameter, beta_2 > 0

        Returns
        -------

        r_kld_gamma: numeric
            Kullback-Liebler Divergence Metric
        
        """

        # Express 
        theta_1 = 1/beta_1
        theta_2 = 1/beta_2
        p1, p2 = 1
        
        # Calculations
        a = p1*(theta_2**alpha_2)*scipy.special.gamma(alpha_2/p2)
        b = p2*(theta_1**alpha_1)*scipy.special.gamma(alpha_1/p1)
        c = (((scipy.special.digamma(alpha_1/p1))/p1) + np.log(theta_1))*(alpha_1 - alpha_2)
        
        # Bi-gamma functions
        d = scipy.special.gamma((alpha_1+p2)/p1)
        e = scipy.special.gamma((alpha_1/p1))
        
        # Calculations
        f = (theta_1/theta_2)**(p2)
        g = alpha_1/p1
        
        # General calculation and output
        r_kld_gamma = np.log(a/b) + c + (d/e)*f - g

        return r_kld_gamma

    # -- KLD result -- #
    # ---------------- #

    if prob_dist == 'gamma':
        
        a_p, b_p = _gamma_params(data=p_data)
        a_q, b_q = _gamma_params(data=q_data)
        
        return _kld_gamma(a_p, b_p, a_q, b_q)

    else:
        print('error')
