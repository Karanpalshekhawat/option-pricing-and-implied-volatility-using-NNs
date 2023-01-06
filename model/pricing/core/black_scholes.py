"""
This module contains methods that
create an option object for a set of
parameters and then compute its BS price
"""

from optionprice import Option


def create_option_object():
    """
    This method creates an option object
    given a set of parameters
    """

    return


def create_dataset(df, s0, input_param):
    """
    This method takes current underlying price and
    a set of other parameters and then computes each
    option BS price

    Args:
        df (pd.DataFrame) : input variables dataframe
        s0 (float) : current underlying price
        input_param (pd.DataFrame) : contains info about money-ness,
                                    time, rfr, volatility to create dataset

    Returns:
         pd.DataFrame
    """
    input_param['s0'] = s0
    input_param['strike'] = input_param.apply(lambda x: x['s0'] / x['moneyness'], axis=1)
    input_param['calender_days'] = input_param.apply(lambda x: x['time_to_maturity'] * 365, axis=1)
    input_param['calender_days'] = input_param['calender_days'].round().astype(int)
    new_column_name = {
        'time_to_maturity': 't', 'strike': 'k',
        'volatility': 'sigma', 'risk_free_rate': 'r',
    }
    input_param.rename(columns=new_column_name, inplace=True)
    input_param['european'] = True if df['opt_type'].iloc[0] == "european" else False
    input_param['kind'] = "call" if df['opt_kind'].iloc[0] == "call" else "put"

    return input_param
