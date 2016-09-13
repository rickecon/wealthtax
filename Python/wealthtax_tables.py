'''
------------------------------------------------------------------------
Last updated 9/9/2016

Creates tables for the wealth tax paper and saves to an excel file.

This py-file calls the following other file(s):

------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import utils
import os
from scipy import stats
import cPickle as pickle

from ogusa import parameters, labor, utils, demographics

baseline_dir = "./OUTPUT_BASELINE"
reform_dir = {}
reform_dir['wealth'] = "./OUTPUT_WEALTH_REFORM"
reform_dir['income'] = "./OUTPUT_INCOME_REFORM"
GRAPH_DIR = './Graphs'


'''
Get parameters
'''
run_params = parameters.get_parameters(baseline=True, reform={},
                      guid={}, user_modifiable=True)
run_params['analytical_mtrs'] = False
S = run_params['S']
E = run_params['E']
T = run_params['T']
starting_age = run_params['starting_age']
ending_age = run_params['ending_age']
J = run_params['J']
lambdas = run_params['lambdas']
l_tilde = run_params['ltilde']
theta = 1/run_params['frisch']
b_ellipse = run_params['b_ellipse']
upsilon = run_params['upsilon']

ss_dir = os.path.join(baseline_dir, "sigma3.0/SS/SS_vars.pkl")
ss_output = pickle.load(open(ss_dir, "rb"))
bssmat_base = ss_output['bssmat']
factor_base = ss_output['factor_ss']
n_base = ss_output['nssmat']
c_base = ss_output['cssmat']
BQ_base = ss_output['BQss']
#utility_base = ss_output[]
#income_base = ss_output[]
Kss_base = ss_output['Kss']
Lss_base = ss_output['Lss']


bssmat = {}
factor = {}
n = {}
c = {}
BQ = {}
for item in ('wealth','income'):
    ss_dir = os.path.join(reform_dir[item], "sigma3.0/SS/SS_vars.pkl")
    ss_output = pickle.load(open(ss_dir, "rb"))
    bssmat[item] = ss_output['bssmat']
    factor[item] = ss_output['factor_ss']
    n[item] = ss_output['nssmat']
    c[item] = ss_output['cssmat']
    BQ[item] = ss_output['BQss']
    #utility[item] = ss_output[]
    #income[item] = ss_output[]


'''
------------------------------------------------------------------------
    Tables
------------------------------------------------------------------------
'''
## Moments, data vs model - plus minstat from GMM estiation (wealth moments only)

## Comparision on changes in the SS gini (total, by age, by type) from baseline
## vs wealth tax and income tax reforms

## Compare changes in aggregate variables in SS - baseline vs two reforms

## Percent changes in alternative inequality measures

## Sensitivity analysis - percentage changes in gini coefficient for wealth and
## income tax with different sigmas



'''
Tables in paper, but that don't need updating with revision:
1) Stationary variable definitions
2) Some others that just define parameters and variables, etc.
'''
