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
import os
from scipy import stats
import cPickle as pickle
import xlsxwriter

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
omega_SS = run_params['omega_SS']
e = run_params['e']

bssmat = {}
factor = {}
n = {}
c = {}
BQ = {}
Kss = {}
Lss = {}
wss = {}
rss = {}
ss_dir = os.path.join(baseline_dir, "sigma3.0/SS/SS_vars.pkl")
ss_output = pickle.load(open(ss_dir, "rb"))
bssmat['base'] = ss_output['bssmat']
factor['base'] = ss_output['factor_ss']
n['base'] = ss_output['nssmat']
c['base'] = ss_output['cssmat']
BQ['base'] = ss_output['BQss']
#utility_base = ss_output[]
#income_base = ss_output[]
Kss['base'] = ss_output['Kss']
Lss['base'] = ss_output['Lss']
wss['base'] = ss_output['wss']
rss['base'] = ss_output['rss']

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
    wss[item] = ss_output['wss']
    rss[item] = ss_output['rss']


'''
------------------------------------------------------------------------
    Tables - all tables for paper saved to different worksheets in an
    Excel workbook
------------------------------------------------------------------------
'''
## open Excel workbook
workbook = xlsxwriter.Workbook('WealthTaxTables.xlsx')

## Moments, data vs model - plus minstat from GMM estimation (wealth moments only)


## Comparision of changes in the SS gini (total, by age, by type) from baseline
## vs wealth tax and income tax reforms
weights = {}
b_dict = {}
y_dict = {}
c_dict = {}
n_dict = {}
gini = {}
weights['total'] = np.tile(omega_SS.reshape(S, 1), (1, J)) * lambdas.reshape(1, J)
weights['ability'] = lambdas
weights['age'] = omega_SS
for tax_run in ('base','wealth','income'):
    income = ((wss[tax_run]*e*n[tax_run]) + (rss[tax_run]*bssmat[tax_run]))*factor['base']
    b_dict[tax_run,'total'] = bssmat[tax_run]
    b_dict[tax_run,'ability'] = bssmat[tax_run].sum(axis=0)
    b_dict[tax_run,'age'] = bssmat[tax_run].sum(axis=1)
    y_dict[tax_run,'total'] = income
    y_dict[tax_run,'ability'] = income.sum(axis=0)
    y_dict[tax_run,'age'] = income.sum(axis=1)
    c_dict[tax_run,'total'] = c[tax_run]
    c_dict[tax_run,'ability'] = c[tax_run].sum(axis=0)
    c_dict[tax_run,'age'] = c[tax_run].sum(axis=1)
    n_dict[tax_run,'total'] = n[tax_run]
    n_dict[tax_run,'ability'] = n[tax_run].sum(axis=0)
    n_dict[tax_run,'age'] = n[tax_run].sum(axis=1)
    for item in ('total','ability','age'):
        gini['b',item,tax_run] = utils.gini(b_dict[tax_run,item], weights[item])
        gini['y',item,tax_run] = utils.gini(y_dict[tax_run,item], weights[item])
        gini['c',item,tax_run] = utils.gini(b_dict[tax_run,item], weights[item])
        gini['n',item,tax_run] = utils.gini(n_dict[tax_run,item], weights[item])

# write to workbook
worksheet = workbook.add_worksheet('Table 3')
row = 2
col = 2
for var in ('b','y','c','n'):
    row+=1
    for item in ('total','ability','age'):
        row+=1
        col = 2
        for tax_run in ('base','wealth','income'):
            worksheet.write(row,col,gini[(var,item,tax_run)])
            col += 2
workbook.close()
quit()



## Compare changes in aggregate variables in SS - baseline vs two reforms

## Percent changes in alternative inequality measures

## Sensitivity analysis - percentage changes in gini coefficient for wealth and
## income tax with different sigmas



'''
Tables in paper, but that don't need updating with revision:
1) Stationary variable definitions
2) Some others that just define parameters and variables, etc.
'''
