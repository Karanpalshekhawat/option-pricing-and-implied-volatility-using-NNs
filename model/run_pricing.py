"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file for pricing options
"""
import pandas as pd
import argparse

from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script')
    parser.add_argument('-h', '--num_dt_hyper', type=int, help='Hyper parameter tuning dataset size')
    parser.add_argument('-t', '--num_dt_training', type=int, help='Actual model run dataset size')
    args = parser.parse_args()
    small_dt_set = pre_processing(args.num_dt_hyper)  # small dataset for hyperparameter tuning
    running_hyperparameter_tuning = False
    if running_hyperparameter_tuning:
        hyperparameter_tuning(small_dt_set)
    file_name = r"./model/output/" + "best_hyper_parameter.pkl"
    df_hyper = pd.read_pickle(file_name)
    big_dataset = pre_processing(args.num_dt_training)
    run_nn_model(big_dataset, df_hyper)
