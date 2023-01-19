"""
This module contains brent method to
compute implied volatility of an option
give heston price and other parameters

Advanatage of using brent method is that
it's derivative free.

Newton raphson method fails as vega is close
to 0 for Deep OTM or ITM options
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


def implied_volatility_call(row):
    """
    This method compute implied volatility
    using brent method which is derivative free

    Args:
        row (pd.DataFrame) : input parameter

    Returns:
        implied volatility in percent
    """
    tol = 0.0001
    max_iterations = 100
    sigma = 0.3  # initial prediction

    param = {
        'strike': row['strike'],
        's0': row['s0'],
        'risk_free_rate': row['risk_free_rate'],
        'calender_days': row['calender_days'],
        'volatility': sigma,
        'european': True,
        'kind': 'call'
    }
    data_param = pd.Series(param)
    for i in range(max_iterations):
        diff = get_bs_price(data_param) - row['Heston_price']
        if abs(diff) < tol:
            break

        sigma = sigma - diff / compute_vega(row['s0'], row['strike'], row['time_to_maturity'], row['risk_free_rate'],
                                            sigma)

    return sigma
