'''
------------------------------------------------------------------------
Last updated 1/29/2015

Returns the wealth for all ages of a certain percentile.

This py-file calls the following other file(s):
            jason_savings_data/scf2007to2013_wealth_age.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Nothing/wealth_data_moments.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
from scipy import stats
import pickle

data = pd.read_table(
    "data/wealth/scf2007to2013_wealth_age_all_percentiles.csv", sep=',', header=0)


def get_highest_wealth_data(bin_weights):
    last_ability_size = bin_weights[-1]
    percentile = 100 - int(last_ability_size * 100)
    highest_wealth_data = np.array(data['p{}_wealth'.format(percentile)])
    var_names = ['highest_wealth_data']
    dictionary = {}
    for key in var_names:
        dictionary[key] = locals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments.pkl", "w"))

