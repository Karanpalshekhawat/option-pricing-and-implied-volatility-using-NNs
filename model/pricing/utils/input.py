"""
In this file, I have added methods that
are used to generate input parameters and
compute intermediary calculations
"""

import json

from scipy.stats import qmc


def read_input_parameters_range():
    """
    Reading input parameters range to create set of
    dataset that will be used for training and testing
    purpose
    """
    pt = r"./model/static_data/"
    json_file = pt + "input-parameters-range.json"
    with open(json_file) as f:
        data = json.load(f)

    return data


def create_set_of_input_parameters(rng):
    """
    Using range of Input parameters defined in json file,
    a set of options parameters range is created using
    Latin hypercube sampling.

    Args:
        rng (dict): input parameters range
    """
    dimension = len(rng)
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler(n=1000)
    l_bounds = [items[0] for items in rng.values()]
    u_bounds = [items[1] for items in rng.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    return sample_scaled
