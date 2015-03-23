'''
------------------------------------------------------------------------
Last updated 3/1/2015

Returns the wealth for all ages of a certain percentile.

This py-file calls the following other file(s):
            data/wealth/scf2007to2013_wealth_age_all_percentiles.csv

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
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_table("data/wealth/scf2007to2013_wealth_age_all_percentiles.csv", sep=',', header=0)
# data = pd.read_table("data/wealth/scf2007to2013_wealth_age.csv", sep=',', header=0)


to_graph = np.array(data)[:, 1:-1]

domain = np.linspace(18, 95, 78)
Jgrid = np.linspace(1, 99, 99)
X, Y = np.meshgrid(domain, Jgrid)
cmap2 = matplotlib.cm.get_cmap('summer')
fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, (to_graph).T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'age-$s$')
ax10.set_ylabel(r'percentile')
ax10.set_zlabel(r'wealth')
plt.savefig('OUTPUT/Demographics/distribution_of_wealth_data')

fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, np.log(to_graph).T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'age-$s$')
ax10.set_ylabel(r'percentile')
ax10.set_zlabel(r'log of wealth')
plt.savefig('OUTPUT/Demographics/distribution_of_wealth_data_log')



def get_highest_wealth_data(bin_weights):
    last_ability_size = bin_weights[-1]
    percentile = 100 - int(last_ability_size * 100)
    highest_wealth_data = np.array(data['p{}_wealth'.format(percentile)])
    var_names = ['highest_wealth_data']
    dictionary = {}
    for key in var_names:
        dictionary[key] = locals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments.pkl", "w"))
