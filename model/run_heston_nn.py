"""
This module runs other modules creates dataset using
lating hypercube sampling and for a particular set of
option parameters, it finds out heston price using
numerical method, implied volatility for that heston price
using brent method and then build a neural network model
learn the non-linear relationship.
"""
import pandas as pd
import argparse

from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take number of data required for running model', add_help=False)
    parser.add_argument('-t', '--num_dt_training', type=int, help='Actual model run dataset size')
    args = parser.parse_args()
    df, st_current_price, range_of_inputs = pre_processing(args.num_dt_training, "HESTON")
    big_dataset = create_heston_dataset(df, st_current_price, range_of_inputs)
    file_name = r"./model/output/" + "best_hyper_parameter.p"
    df_hyper = pd.read_pickle(file_name)
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'correlation', 'reversion_speed',
                       'Long_average_variance', 'vol_vol', 'initial_variance']
    target = 'opt_price_by_strike'
    model = run_nn_model(big_dataset, df_hyper, feature_columns, target)
    model_save_path = r"./model/output/" + "Heston_NN_model.h5"
    model.save(model_save_path)
