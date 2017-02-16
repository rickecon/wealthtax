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

from ogusa import parameters, labor, utils, inequal, demographics

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
Yss = {}
Kss = {}
Lss = {}
wss = {}
rss = {}
T_Hss = {}
ss_dir = os.path.join(baseline_dir, "sigma2.0/SS/SS_vars.pkl")
ss_output = pickle.load(open(ss_dir, "rb"))
bssmat['base'] = ss_output['bssmat']
factor['base'] = ss_output['factor_ss']
n['base'] = ss_output['nssmat']
c['base'] = ss_output['cssmat']
BQ['base'] = ss_output['BQss']
Yss['base'] = ss_output['Yss']
Kss['base'] = ss_output['Kss']
Lss['base'] = ss_output['Lss']
wss['base'] = ss_output['wss']
rss['base'] = ss_output['rss']
T_Hss['base'] = ss_output['T_Hss']

for item in ('wealth','income'):
    ss_dir = os.path.join(reform_dir[item], "sigma2.0/SS/SS_vars.pkl")
    ss_output = pickle.load(open(ss_dir, "rb"))
    bssmat[item] = ss_output['bssmat']
    factor[item] = ss_output['factor_ss']
    n[item] = ss_output['nssmat']
    c[item] = ss_output['cssmat']
    BQ[item] = ss_output['BQss']
    Yss[item] = ss_output['Yss']
    Kss[item] = ss_output['Kss']
    Lss[item] = ss_output['Lss']
    wss[item] = ss_output['wss']
    rss[item] = ss_output['rss']
    T_Hss[item] = ss_output['T_Hss']


'''
------------------------------------------------------------------------
    Tables - all tables for paper saved to different worksheets in an
    Excel workbook
------------------------------------------------------------------------
'''
## open Excel workbook
workbook = xlsxwriter.Workbook('WealthTaxTables.xlsx')

'''
Moments, data vs model - plus minstat from GMM estimation (wealth moments only)
'''
# read in data saved from calibration
mom_dir = os.path.join(baseline_dir, "Calibration/moment_results.pkl")
moment_fit = pickle.load(open(mom_dir, "rb"))

# write to workbook
worksheet = workbook.add_worksheet('Moments')
worksheet.write(0,0,'Moment')
worksheet.write(0,1,'Data')
worksheet.write(0,2,'Model')
moment_names=['Share 0-25%','Share 25-50%','Share 50-70%','Share 70-80%',
             'Share 80-90%','Share 90-99%','Share 99-100%','Gini Coefficient',
             'var(ln(wealth))']
row = 1
for i in range(len(moment_names)):
    col = 0
    worksheet.write(row,col,moment_names[i])
    col+=1
    worksheet.write(row,col,moment_fit['data_moment'][i])
    col+=1
    worksheet.write(row,col,moment_fit['model_moment'][i])
    row+=1
worksheet.write(row,0,'Minimum Statistic')
worksheet.write(row,col,moment_fit['minstat'].loc[0])



'''
Comparision of changes in the SS gini (total, by age, by type) from baseline
vs wealth tax and income tax reforms
'''
weights = {}
b_dict = {}
y_dict = {}
c_dict = {}
n_dict = {}
gini = {}
weights['Total'] = np.tile(omega_SS.reshape(S, 1), (1, J)) * lambdas.reshape(1, J)
weights['Ability $j$'] = lambdas
weights['Age $s$'] = omega_SS
for tax_run in ('base','wealth','income'):
    income = ((wss[tax_run]*e*n[tax_run]) + (rss[tax_run]*bssmat[tax_run]))*factor['base']
    b_dict[tax_run,'Total'] = bssmat[tax_run]
    b_dict[tax_run,'Ability $j$'] = bssmat[tax_run].sum(axis=0)
    b_dict[tax_run,'Age $s$'] = bssmat[tax_run].sum(axis=1)
    y_dict[tax_run,'Total'] = income
    y_dict[tax_run,'Ability $j$'] = income.sum(axis=0)
    y_dict[tax_run,'Age $s$'] = income.sum(axis=1)
    c_dict[tax_run,'Total'] = c[tax_run]
    c_dict[tax_run,'Ability $j$'] = c[tax_run].sum(axis=0)
    c_dict[tax_run,'Age $s$'] = c[tax_run].sum(axis=1)
    n_dict[tax_run,'Total'] = n[tax_run]
    n_dict[tax_run,'Ability $j$'] = n[tax_run].sum(axis=0)
    n_dict[tax_run,'Age $s$'] = n[tax_run].sum(axis=1)
    for item in ('Total','Ability $j$','Age $s$'):
        gini['b',item,tax_run] = inequal.gini(b_dict[tax_run,item], weights[item])
        gini['y',item,tax_run] = inequal.gini(y_dict[tax_run,item], weights[item])
        gini['c',item,tax_run] = inequal.gini(c_dict[tax_run,item], weights[item])
        gini['n',item,tax_run] = inequal.gini(n_dict[tax_run,item], weights[item])

# write to workbook
worksheet = workbook.add_worksheet('Gini Changes')
top_line = ['Wealth Tax', 'Income Tax']
headings = ['Steady-State Variable', 'Gini Type','Baseline','Treatment','% Change','Treatment','% Change']
tex_vars = ['$\\bar{b}_{j,s}$','$\\bar{y}_{j,s}$','$\\bar{c}_{j,s}$','$\\bar{n}_{j,s}$']
vars_names = ['Wealth','Income','$Consumption','Labor Supply']
row = 1
col = 0
for item in headings:
    worksheet.write(row,col,item)
    col+=1
worksheet.merge_range('D1:E1', top_line[0])
worksheet.merge_range('F1:G1', top_line[1])

col = 0
row = 2
for i in range(len(tex_vars)):
    worksheet.write(row,col,tex_vars[i])
    row+=1
    worksheet.write(row,col,vars_names[i])
    row+=2
row = 2
col = 1
for var in ('b','y','c','n'):
    for item in ('Total','Ability $j$','Age $s$'):
        col=1
        worksheet.write(row,col,item)
        col=2
        worksheet.write(row,col,gini[(var,item,'base')])
        col=3
        for tax_run in ('wealth','income'):
            worksheet.write(row,col,gini[(var,item,tax_run)])
            col += 1
            pct_diff = (gini[(var,item,tax_run)]-gini[(var,item,'base')])/gini[(var,item,tax_run)]
            worksheet.write(row,col,pct_diff)
            col += 1
        row+=1



'''
Compare changes in aggregate variables in SS - baseline vs two reforms
'''
# Create dict for agg vars
agg_dict = {}
for tax_run in ('base','wealth','income'):
    agg_dict['Yss',tax_run] = Yss[tax_run]
    agg_dict['Kss',tax_run] = Kss[tax_run]
    agg_dict['Lss',tax_run] = Lss[tax_run]
    agg_dict['Css',tax_run] = (c[tax_run]*(np.tile(omega_SS.reshape(S, 1), (1, J))
                                           * lambdas.reshape(1, J))).sum()
    #u = ((c[tax_run]**(1-sigma))/(1-sigma)) + np.exp(g_y_ss*(1-sigma))*chi_n
    u = 50
    agg_dict['Uss',tax_run] = (u*(np.tile(omega_SS.reshape(S, 1), (1, J))
                                           * lambdas.reshape(1, J))).sum()

# write to workbook
worksheet = workbook.add_worksheet('Aggregate Changes')
top_line = ['Wealth Tax', 'Income Tax']
headings = ['Steady-State Aggregate Variable','Baseline','Treatment','% Change','Treatment','% Change']
row_labels = ['Income (GDP) $\\bar{Y}', 'Capital Stock $\\bar{K}$', 'Labor $\\bar{L}$',
              'Consumption $\\bar{C}*$', 'Total Utility $\\bar{U}*$']
var_list  = ['Yss','Kss','Lss','Css','Uss']
row = 1
col = 0
for item in headings:
    worksheet.write(row,col,item)
    col+=1
worksheet.merge_range('D1:E1', top_line[0])
worksheet.merge_range('F1:G1', top_line[1])

row = 2
for i in range(len(row_labels)):
    col = 0
    worksheet.write(row,col,row_labels[i])
    col+=1
    worksheet.write(row,col,agg_dict[var_list[i],'base'])
    col+=1
    for tax_run in ('wealth','income'):
        worksheet.write(row,col,agg_dict[var_list[i],tax_run])
        col += 1
        pct_diff = ((agg_dict[var_list[i],tax_run]-
                     agg_dict[var_list[i],'base'])/agg_dict[var_list[i],tax_run])
        worksheet.write(row,col,pct_diff)
        col += 1
    row+=1



'''
Percent changes in alternative inequality measures
'''
inequality = {}
weights = np.tile(omega_SS.reshape(S, 1), (1, J)) * lambdas.reshape(1, J)
for tax_run in ('base','wealth','income'):
    income = ((wss[tax_run]*e*n[tax_run]) + (rss[tax_run]*bssmat[tax_run]))
    var_dict = {'b':bssmat[tax_run],'y':income,'c':c[tax_run],'n':n[tax_run]}
    for key,value in var_dict.iteritems():
        inequality[key,tax_run,'$var(log(x_{j,s}))$'] = inequal.var_log(value, weights,factor['base'])
        inequality[key,tax_run,'90/10 ratio'] = inequal.ninety_ten(value, weights)
        inequality[key,tax_run,'Top 10% share'] = inequal.top_10(value, weights)
        inequality[key,tax_run,'Top 1% share'] = inequal.top_1(value, weights)

# write to workbook
worksheet = workbook.add_worksheet('Alt Inequality')
top_line = ['Wealth Tax', 'Income Tax']
headings = ['Steady-state Variable', 'Inequality Measure','Baseline Value','Treatment','% Change','Treatment','% Change']
tex_vars = ['$\\bar{b}_{j,s}$','$\\bar{y}_{j,s}$','$\\bar{c}_{j,s}$']
vars_names = ['Wealth','Income','$Consumption']
ineq_measures = ['$var(log(x_{j,s}))$','90/10 ratio','Top 10% share','Top 1% share']
row = 1
col = 0
for item in headings:
    worksheet.write(row,col,item)
    col+=1
worksheet.merge_range('D1:E1', top_line[0])
worksheet.merge_range('F1:G1', top_line[1])

col = 0
row = 2
for i in range(len(tex_vars)):
    worksheet.write(row,col,tex_vars[i])
    row+=1
    worksheet.write(row,col,vars_names[i])
    row+=3
row = 2
col = 1
for var in ('b','y','c'):
    for item in ineq_measures:
        col=1
        worksheet.write(row,col,item)
        col=2
        worksheet.write(row,col,inequality[(var,'base',item)])
        col=3
        for tax_run in ('wealth','income'):
            worksheet.write(row,col,inequality[(var,tax_run,item)])
            col += 1
            pct_diff = (inequality[(var,tax_run,item)]-inequality[(var,'base',item)])/inequality[(var,tax_run,item)]
            worksheet.write(row,col,pct_diff)
            col += 1
        row+=1




'''
Sensitivity analysis - percentage changes in gini coefficient for wealth and
income tax with different sigmas
'''
weights = np.tile(omega_SS.reshape(S, 1), (1, J)) * lambdas.reshape(1, J)
sigma_list = [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 3.2]
dir_dict = {}
gini = {}
for sig_val in sigma_list:
    dir_dict['base'] = "./OUTPUT_BASELINE" + '/sigma' + str(sig_val)
    dir_dict['wealth'] = "./OUTPUT_WEALTH_REFORM" + '/sigma' + str(sig_val)
    dir_dict['income'] = "./OUTPUT_INCOME_REFORM" + '/sigma' + str(sig_val)
    for item in ('base','wealth','income'):
        #ss_dir = os.path.join(dir_dict[item], "/SS/SS_vars.pkl")
        ss_dir = dir_dict[item]+ '/SS/SS_vars.pkl'
        ss_output = pickle.load(open(ss_dir, "rb"))
        bss = ss_output['bssmat']
        nss = ss_output['nssmat']
        css = ss_output['cssmat']
        wss = ss_output['wss']
        rss = ss_output['rss']
        income = ((wss*e*nss) + (rss*bss))*factor['base']
        gini['b',item,str(sig_val)] = np.nan_to_num(inequal.gini(bss, weights))
        gini['y',item,str(sig_val)] = np.nan_to_num(inequal.gini(income, weights))
        gini['c',item,str(sig_val)] = np.nan_to_num(inequal.gini(css, weights))
        gini['n',item,str(sig_val)] = np.nan_to_num(inequal.gini(nss, weights))

# write to workbook
worksheet = workbook.add_worksheet('Robust Sigma')
top_line = ['Wealth Tax', 'Income Tax']
headings = ['Steady-State Variable', 'CRRA','Baseline','Treatment','% Change','Treatment','% Change']
tex_vars = ['$\\bar{b}_{j,s}$','$\\bar{y}_{j,s}$','$\\bar{c}_{j,s}$','$\\bar{n}_{j,s}$']
vars_names = ['Wealth','Income','$Consumption','Labor Supply']
row = 1
col = 0
for item in headings:
    worksheet.write(row,col,item)
    col+=1
worksheet.merge_range('D1:E1', top_line[0])
worksheet.merge_range('F1:G1', top_line[1])

col = 0
row = 2
for i in range(len(tex_vars)):
    worksheet.write(row,col,tex_vars[i])
    row+=1
    worksheet.write(row,col,vars_names[i])
    row+=3
row = 2
col = 1
for var in ('b','y','c','n'):
    for sig_val in sigma_list:
        col=1
        worksheet.write(row,col,'$\sigma='+str(sig_val))
        col=2
        worksheet.write(row,col,gini[(var,'base',str(sig_val))])
        col=3
        for tax_run in ('wealth','income'):
            worksheet.write(row,col,gini[(var,tax_run,str(sig_val))])
            col += 1
            pct_diff = np.nan_to_num((gini[(var,tax_run,str(sig_val))]-
                        gini[(var,'base',str(sig_val))])/gini[(var,tax_run,str(sig_val))])
            worksheet.write(row,col,pct_diff)
            col += 1
        row+=1


workbook.close()



'''
Tables in paper, but that don't need updating with revision:
1) Stationary variable definitions
2) Some others that just define parameters and variables, etc.
'''
