"""
This module contains the model definition
and a method to identify the best learning rate
"""

import tensorflow as tf

def nn_model(row, features):
    """
    Just defines the model structure

    Args:
        row (pd.DataFrame) : hyperparameters
        features (int): number of features

    Return:
        Sequential
    """
    model_ind = tf.keras.Sequential()
    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization'],
                                        input_shape=(features.shape[1],)))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization']))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization']))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dense(1))

    return model_ind
