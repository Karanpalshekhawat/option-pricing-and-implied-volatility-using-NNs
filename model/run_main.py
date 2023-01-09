"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file
"""

from model import *


def pre_processing():
    """
    This method creates the dataset that we will
    use for training and testing the NN model

    Returns:
        pd.DataFrame
    """
    df = read_input_variables_file()
    ticker = df['ticker'].iloc[0]
    st_current_price = get_current_price(ticker)
    range_of_inputs = create_set_of_input_parameters()
    option_df = create_dataset(df, st_current_price, range_of_inputs)

    return option_df


def run_model(dt_set):
    """
    This method split data set into train, test and then
    creates model, finds the best possible hyperparameter,
    stores them to a file and then predict its accuracy on
    the test set.

    Args:
        dt_set (pd.DataFrame) : dataset
    """
    nn_obj = create_nn_architecture(dt_set)
    return


if __name__ == "__main__":
    dt_set = pre_processing()
    nn_model = run_model(dt_set)
