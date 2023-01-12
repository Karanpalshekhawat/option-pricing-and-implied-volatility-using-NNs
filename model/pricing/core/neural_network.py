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


def create_nn_architecture(input_features):
    """
    This method creates the neural network
    architecture

    Args:
        input_features (EagerTensor) : tensor of input features

    Returns:
        Sequential
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(400, activation="relu", kernel_initializer='he_normal',
                              input_shape=(input_features.shape[1],)),
        tf.keras.layers.Dense(400, activation="relu", kernel_initializer='he_normal'),
        tf.keras.layers.Dense(400, activation="relu", kernel_initializer='he_normal'),
        tf.keras.layers.Dense(1)
    ])
    return model


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

    return pd.DataFrame(combinations, columns=[str(f) for f in ls_param])


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
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'volatility']
    input_features = dt_set[feature_columns]
    target = dt_set['opt_price_by_strike']
    x_train, x_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2)
    nn_obj = create_nn_architecture(input_features)
    nn_obj.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    nn_obj.fit(x_train, y_train, epochs=5)
    mse = nn_obj.evaluate(x_test, y_test)
    return nn_obj
