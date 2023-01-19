"""
This module runs other modules which creates small
dataset for hyperparameter tuning, find out heston
model option pricing using numerical method, and then
build a neural network model to compute it.
"""
import pandas as pd
import argparse

from model import *

if __name__ == "__main__":
    n_hyper = 100000
    df, st_current_price, range_of_inputs = pre_processing(n_hyper, "HESTON")  # small dataset for hyperparameter tuning
    small_dt_set = create_heston_dataset(df, st_current_price, range_of_inputs)
