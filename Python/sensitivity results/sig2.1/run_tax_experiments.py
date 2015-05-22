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
scal = np.ones(J) * 1.1
scal[-1] = .5
scal[-2] = .7

SS_stage = 'SS_tax'

chi_b_scal = np.zeros(J)

h_wealth = 0.297913669300355
m_wealth = 1.91632477396983

d_tax_income = .219


print 'Getting SS distribution for wealth tax.'
var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'chi_b_scal', 'SS_stage']
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

# call(['python', 'TPI.py'])


shutil.rmtree('OUTPUT_wealth_tax')
shutil.copytree('OUTPUT', 'OUTPUT_wealth_tax')

# '''
# Run Steady State Solver and TPI for wealth tax
# '''
p_wealth = 0.0
scal = np.ones(J)

var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'chi_b_scal', 'SS_stage']

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
    print 'Error in taxes:', error
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
             'a_tax_income', 'scal',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'chi_b_scal', 'SS_stage']

dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
print 'Getting SS distribution for income tax.'
call(['python', 'SS.py'])

# call(['python', 'TPI.py'])

shutil.rmtree('OUTPUT_income_tax')
shutil.copytree('OUTPUT', 'OUTPUT_income_tax')

files = glob('*.pyc')
for i in files:
    os.remove(i)
