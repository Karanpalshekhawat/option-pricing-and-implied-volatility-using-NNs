"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file
"""

import datetime

from pandas.tseries.offsets import BDay

if __name__ == "__main__":
    date = datetime.datetime.today()
    bz_day = date - BDay(-1)
