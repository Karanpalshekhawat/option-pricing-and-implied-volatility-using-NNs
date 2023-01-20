"""
This module contains brent method to
compute implied volatility of an option
give heston price and other parameters

Advantage of using brent method is that
it's derivative free.

Newton raphson method fails as vega is close
to 0 for Deep OTM or ITM options
"""

import pandas as pd

from scipy.optimize import brentq
from model.pricing.core.black_scholes import get_bs_price


def implied_volatility_call(row):
    """
    This method compute implied volatility
    using brent method which is derivative free

    Args:
        row (pd.DataFrame) : input parameter

    Returns:
        implied volatility in percent
    """

    def difference(sigma):
        row['volatility'] = sigma
        return row['Heston_price'] - get_bs_price(row)

    return brentq(difference, 0.001, 10, maxiter=1000)
