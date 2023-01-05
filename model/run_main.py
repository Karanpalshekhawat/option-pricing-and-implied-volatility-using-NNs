"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file
"""

import datetime

from model import *

if __name__ == "__main__":
    date = datetime.datetime.today()
    df = read_input_variables_file()
    ticker = df['ticker'].iloc[0]
    st_current_price = get_current_price(ticker)
    range_of_inputs = create_set_of_input_parameters()
