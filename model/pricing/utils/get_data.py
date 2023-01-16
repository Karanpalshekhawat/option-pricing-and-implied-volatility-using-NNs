"""
This module contains different methods
which retrieve either main or intermediary
data
"""

import yfinance as yf

from model.pricing.utils.input import read_input_variables_file, create_set_of_input_parameters
from model.pricing.core.black_scholes import create_dataset


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


def pre_processing(nmb):
    """
    This method creates the dataset that we will
    use for training and testing the NN model

    Args:
        nmb (int) : number of dataset to generate

    Returns:
        pd.DataFrame
    """
    df = read_input_variables_file()
    ticker = df['ticker'].iloc[0]
    st_current_price = get_current_price(ticker)
    range_of_inputs = create_set_of_input_parameters(nmb)
    option_df = create_dataset(df, st_current_price, range_of_inputs)

    return option_df
