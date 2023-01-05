"""
This module contains different methods
which retrieve either main or intermediary
data
"""

import yfinance as yf


def get_current_price(ticker):
    """
    This method retrieves price as of last business day
    for date and ticker

    Args:
        ticker (str): symbol applicable for yfinance

    Returns:
        float
    """
    dt = yf.Ticker(ticker)
    df = dt.history()

    return df['Close'].iloc[0]
