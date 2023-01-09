"""
This module runs the main neural network model
and also tune the hyperparameter, store them to
a output file and test accuracy
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split


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
        tf.keras.layers.Input(shape=(input_features.shape[1],)),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model


def run_model(dt_set):
    """
    This method split data set into train, test and then
    creates model, finds the best possible hyperparameter,
    stores them to a file and then predict its accuracy on
    the test set.

    Args:
        dt_set (pd.DataFrame) : dataset
    """
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'volatility']
    input_features = tf.convert_to_tensor(dt_set[feature_columns])
    target = tf.convert_to_tensor(dt_set.pop('opt_price_by_strike'))
    nn_obj = create_nn_architecture(input_features)
    """
    Loss function is the one where the model is trained but
    you can always define multiple metrics where you want to
    measure the performance of the model
    """
    nn_obj.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    x_train, x_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2)
    return nn_obj
