"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file
"""

from model import *


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


if __name__ == "__main__":
    dt_set_hyper = pre_processing(100000)  # for hyperparameter training
    best_hyper_parameter = run_model(dt_set_hyper)
