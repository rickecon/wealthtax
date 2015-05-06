'''
------------------------------------------------------------------------
Last updated 3/17/2015

This will run the steady state solver as well as time path iteration,
given that these have already run with run_model.py, with new tax
policies (calibrating the income tax to match the wealth tax).
------------------------------------------------------------------------
'''

'''
Import Packages
'''

import numpy as np
import pickle
from glob import glob
import os
import sys
import scipy.optimize as opt
import shutil
from subprocess import call


# Import Parameters from initial simulations
variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

# New Tax Parameters
p_wealth = 0.025
SS_initial_run = False
name_of_last = 'initial_guesses_for_SS'
thetas_simulation = False
scal[-1] = .5
name_of_it = 'initial_guesses_for_SS'

chi_b_scaler = False
chi_b_scal = np.zeros(J)

h_wealth = 0.277470036398397
m_wealth = 2.40486776796377

print 'Getting SS distribution for wealth tax.'
var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal', 'thetas_simulation',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump', 'name_of_it',
             'theta_tax', 'retire', 'mean_income', 'name_of_last',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run', 'chi_b_scal', 'chi_b_scaler']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))



'''
Run steady state solver and TPI (according to given variables) for wealth tax
'''
call(['python', 'SS.py'])


TPI_initial_run = False
var_names = ['TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/tpi_var.pkl", "w"))

call(['python', 'TPI.py'])


shutil.rmtree('OUTPUT_wealth_tax')
shutil.copytree('OUTPUT', 'OUTPUT_wealth_tax')

'''
Run Steady State Solver and TPI for wealth tax
'''
p_wealth = 0.0

var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal', 'thetas_simulation',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump', 'name_of_it',
             'theta_tax', 'retire', 'mean_income', 'name_of_last',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run', 'chi_b_scal', 'chi_b_scaler']

dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))

lump_to_match = pickle.load(open("OUTPUT/SS/Tss_var.pkl", "r"))


def matcher(d_inc_guess):
    pickle.dump(d_inc_guess, open("OUTPUT/SS/d_inc_guess.pkl", "w"))
    call(['python', 'SS.py'])
    lump_new = pickle.load(open("OUTPUT/SS/Tss_var.pkl", "r"))
    error = abs(lump_to_match - lump_new)
    # print error
    return error

print 'Computing new income tax to match wealth tax'
new_d_inc = opt.fsolve(matcher, d_tax_income, xtol=1e-13)
print '\tOld income tax:', d_tax_income
print '\tNew income tax:', new_d_inc

os.remove("OUTPUT/SS/d_inc_guess.pkl")
os.remove("OUTPUT/SS/Tss_var.pkl")

d_tax_income = new_d_inc

var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal', 'thetas_simulation',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump', 'name_of_it',
             'theta_tax', 'retire', 'mean_income', 'name_of_last',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run', 'chi_b_scal', 'chi_b_scaler']

dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
print 'Getting SS distribution for income tax.'
call(['python', 'SS.py'])

call(['python', 'TPI.py'])

shutil.rmtree('OUTPUT_income_tax')
shutil.copytree('OUTPUT', 'OUTPUT_income_tax')

files = glob('*.pyc')
for i in files:
    os.remove(i)
