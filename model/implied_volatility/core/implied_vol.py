"""
This module contains newton raphson numerical
method to compute implied volatility of an option
give heston price and other parameters
"""

import numpy as np
import pandas as pd

from scipy.stats import norm
from model.pricing.core.black_scholes import get_bs_price


def compute_vega(S, K, T, r, sigma):
    """
    This method computes vega of an option using
    analytical BS formula.

    Args:
        S: Asset price
        K: Strike price
        T: Time to Maturity
        r: risk-free rate
        sigma : volatility

    Returns:
        partial derivative w.r.t volatility
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    vega = S * np.sqrt(T) * norm.pdf(d1)
    return vega


def implied_volatility_call(C, S, K, T, r):
    """
    This method compute implied volatility
    using newton raphson method

    Args:
        C: Observed call price
        S: Asset price
        K: Strike price
        T: Time to Maturity
        r: risk-free rate

    Returns:
        implied volatility in percent
    """
    tol = 0.0001
    max_iterations = 100
    sigma = 0.3  # initial prediction

    param = {
        'strike': K,
        's0': S,
        'risk_free_rate': r,
        'calender_days': T * 365,
        'volatility': sigma,
        'european': True,
        'kind': 'call'
    }
    data_param = pd.Series(param)
    for i in range(max_iterations):
        diff = get_bs_price(data_param) - C
        if abs(diff) < tol:
            break

        sigma = sigma - diff / compute_vega(S, K, T, r, sigma)

    return sigma
