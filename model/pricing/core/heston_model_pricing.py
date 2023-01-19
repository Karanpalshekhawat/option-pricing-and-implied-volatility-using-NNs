"""
This module contains numerical method approach
to determine heston model price for a set of
parameters
"""

import numpy as np


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, steps, Npaths):
    """
    Generates simultaneous simulation of stock price and
    volatility process to solve for final price using numerical method

    Args:
        S : initial stock price
        T : time to maturity
        r : risk-free rate
        kappa : volatility mean reverting speed
        theta : long term variance
        v_0 : initial variance
        rho : correlation coefficient between wiener processes of stock and volatility evolution
        xi : vol-vol or volatility of the variance
        steps : numbers of steps in dt
        Npaths : number of simulation meaning number of experiments to find final price
    """
    dt = T / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    s_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]),
                                           size=Npaths) * np.sqrt(dt)

        s_t = s_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = s_t

    return prices
