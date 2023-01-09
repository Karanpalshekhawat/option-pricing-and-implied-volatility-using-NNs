"""
This module runs the main neural network model
and also tune the hyperparameter, store them to
a output file and test accuracy
"""

import tensorflow as tf


def create_nn_architecture(input):
    """
    This method creates the neural network
    architecture

    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input.shape[1],)),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model
