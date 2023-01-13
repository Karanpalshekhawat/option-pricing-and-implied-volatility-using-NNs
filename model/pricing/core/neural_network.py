"""
This module runs the main neural network model
and also tune the hyperparameter, store them to
a output file and test accuracy
"""

import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.pricing.utils.input import read_hyper_parameters_range


def find_best_hyper_parameter_config(df_hyper_param, dt_set):
    """
    This method creates the neural network
    architecture for a particular set of hyperparameter

    Args:
        df_hyper_param (pd.DataFrame) : multiple choices for hyperparameter selection
        dt_set (pd.DataFrame) : full dataset

    Returns:
        Sequential
    """
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'volatility']
    input_features = dt_set[feature_columns]
    target = dt_set['opt_price_by_strike']
    x_train, x_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2)

    def model_define_and_evaluate(row):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(row['neurons'], activation=row['activation'], kernel_initializer=row['initialization'],
                                  input_shape=(input_features.shape[1],)),
            tf.keras.layers.Dense(row['neurons'], activation=row['activation'], kernel_initializer=row['initialization']),
            tf.keras.layers.Dense(row['neurons'], activation=row['activation'], kernel_initializer=row['initialization']),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=row['optimizer'], loss='mean_squared_error', metrics=['mse'])
        model.fit(x_train, y_train, epochs=20, batch_size=row['batch_size'], verbose=0)
        mse = model.evaluate(x_test, y_test)

        return model, mse

    for length in range(len(df_hyper_param)):
        row = df_hyper_param.iloc[length]
        model, test_score = model_define_and_evaluate(row)
        if length == 0:
            metric_track = test_score
            final_model = model
        if test_score <= metric_track:
            final_model = model
            metric_track = test_score

    return final_model


def create_set_of_hyperparameter():
    """
    This method first reads the hyperparameter range
    and then creates a dataframe of hyperparameter
    with each row contains unique set of hyperparameter

    Returns:
         pd.DataFrame
    """
    hyper_parameters = read_hyper_parameters_range()
    activation_func = hyper_parameters['activation']
    neuron_list = list(np.arange(hyper_parameters['neurons'][0], hyper_parameters['neurons'][1], 100))
    drop_out_rate = hyper_parameters['dropout_rate']
    initialization_param = hyper_parameters['initialization']
    batch_normalisation = hyper_parameters['batch_normalisation']
    optimizer = hyper_parameters['optimizer']
    batch_size = list(np.arange(hyper_parameters['batch_size'][0], hyper_parameters['batch_size'][1], 256))

    ls_param = [activation_func, neuron_list, drop_out_rate, initialization_param, batch_normalisation, optimizer,
                batch_size]
    combinations = list(itertools.product(*ls_param))
    columns_name = ['activation', 'neurons', 'drop_out', 'initialization', 'batch_normalisation', 'optimizer',
                    'batch_size']
    return pd.DataFrame(combinations, columns=columns_name)


def run_model(dt_set):
    """
    This method split data set into train, test and then
    creates model, finds the best possible hyperparameter,
    stores them to a file and then predict its accuracy on
    the test set.

    Loss function is the one where the model is trained but
    you can always define multiple metrics where you want to
    track the performance of the model

    Args:
        dt_set (pd.DataFrame) : dataset
    """
    df_hyper_param = create_set_of_hyperparameter()
    nn_objs = find_best_hyper_parameter_config(df_hyper_param, dt_set)

    return nn_objs
