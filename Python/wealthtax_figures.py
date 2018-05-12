'''
------------------------------------------------------------------------
Last updated 9/9/2016

Creates figures for the wealth tax paper and saves to an excel file.

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
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ogusa import parameters, labor, utils, demographics, inequal

baseline_dir = "./OUTPUT_BASELINE"
reform_dir = {}
reform_dir['base'] = "./OUTPUT_BASELINE"
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
omega = run_params['omega']
l_tilde = run_params['ltilde']
chi_n_vec = run_params['chi_n_guess']
theta = 1/run_params['frisch']
b_ellipse = run_params['b_ellipse']
upsilon = run_params['upsilon']

ss_dir = os.path.join(baseline_dir, "sigma2.0/SS/SS_vars.pkl")
ss_output = pickle.load(open(ss_dir, "rb"))
bssmat_base = ss_output['bssmat']
factor_base = ss_output['factor_ss']
n_base = ss_output['nssmat']
c_base = ss_output['cssmat']
BQ_base = ss_output['BQss']
#utility_base = ss_output[]
income_at_base = ss_output['y_aftertax_ss']
Kss_base = ss_output['Kss']
Lss_base = ss_output['Lss']
Css_base = ss_output['Css']
Iss_base = ss_output['Iss']
T_Hss_base = ss_output['T_Hss']
Gss_base = ss_output['Gss']
rss_base = ss_output['rss']
wss_base = ss_output['wss']


bssmat = {}
factor = {}
n = {}
c = {}
BQ = {}
T_Hss = {}
Gss = {}
Kss = {}
Lss = {}
income_at = {}
rss = {}
wss = {}
rpath = {}
wpath = {}
Kpath = {}
Lpath = {}
Cpath = {}
Ipath = {}
T_Hpath = {}
Gpath = {}
gini_path = {}
weights = {}
weights['Total'] = np.tile(omega.reshape(T+S, S, 1), (1, 1, J)) * lambdas.reshape(1, J)
weights['Ability $j$'] = np.tile(np.reshape(lambdas, (1, J)), (T + S, 1))
weights['Age $s$'] = omega  # T+S x S in dimension
tpi_names = {'b': 'b_mat', 'b_at': 'b_aftertax_path', 'y': 'y_path',
             'y_at': 'y_aftertax_path', 'c': 'c_path', 'n': 'n_mat'}
for item in ('base', 'wealth', 'income'):
    ss_dir = os.path.join(reform_dir[item], "sigma2.0/SS/SS_vars.pkl")
    tpi_dir = os.path.join(reform_dir[item], "sigma2.0/TPI/TPI_vars.pkl")
    ss_output = pickle.load(open(ss_dir, "rb"))
    tpi_output = pickle.load(open(tpi_dir, "rb"))
    bssmat[item] = ss_output['bssmat']
    factor[item] = ss_output['factor_ss']
    n[item] = ss_output['nssmat']
    c[item] = ss_output['cssmat']
    BQ[item] = ss_output['BQss']
    T_Hss[item] = ss_output['T_Hss']
    Gss[item] = ss_output['Gss']
    Kss[item] = ss_output['Kss']
    Lss[item] = ss_output['Lss']
    #utility[item] = ss_output[]
    income_at[item] = ss_output['y_aftertax_ss']
    rss[item] = ss_output['rss']
    wss[item] = ss_output['wss']
    rpath[item] = tpi_output['r']
    wpath[item] = tpi_output['w']
    Kpath[item] = tpi_output['K']
    Lpath[item] = tpi_output['L']
    Cpath[item] = tpi_output['C']
    Ipath[item] = tpi_output['I']
    T_Hpath[item] = tpi_output['T_H']
    Gpath[item] = tpi_output['G']


    # b_dict[tax_run,'Total'] = b_aftertax_ss[tax_run]
    # b_dict[tax_run,'Ability $j$'] = b_aftertax_ss[tax_run].sum(axis=0)
    # b_dict[tax_run,'Age $s$'] = b_aftertax_ss[tax_run].sum(axis=1)
    # y_dict[tax_run,'Total'] = income
    # y_dict[tax_run,'Ability $j$'] = income.sum(axis=0)
    # y_dict[tax_run,'Age $s$'] = income.sum(axis=1)
    # c_dict[tax_run,'Total'] = c[tax_run]
    # c_dict[tax_run,'Ability $j$'] = c[tax_run].sum(axis=0)
    # c_dict[tax_run,'Age $s$'] = c[tax_run].sum(axis=1)
    # n_dict[tax_run,'Total'] = n[tax_run]
    # n_dict[tax_run,'Ability $j$'] = n[tax_run].sum(axis=0)
    # n_dict[tax_run,'Age $s$'] = n[tax_run].sum(axis=1)

    for var_name in ('b', 'b_at', 'y', 'y_at', 'c', 'n'):
        gini_path[var_name, 'Total', item] = np.zeros_like(tpi_output['K'])
        gini_path[var_name, 'Ability $j$', item] = np.zeros_like(tpi_output['K'])
        gini_path[var_name, 'Age $s$', item] = np.zeros_like(tpi_output['K'])
        for t in range(T):
            gini_path[var_name, 'Total', item][t] = inequal.gini(tpi_output[tpi_names[var_name]][t, :, :], weights['Total'][t, ...])
            gini_path[var_name, 'Ability $j$', item][t] = inequal.gini(tpi_output[tpi_names[var_name]][t, :, :].sum(axis=0), weights['Ability $j$'][t, ...])
            gini_path[var_name, 'Age $s$', item][t] = inequal.gini(tpi_output[tpi_names[var_name]][t, :, :].sum(axis=1), weights['Age $s$'][t, ...])
    # for gini_type in ('Total', 'Ability $j$', 'Age $s$'):
    #     gini_path['b', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     gini_path['b_at', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     gini_path['y', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     gini_path['y_at', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     gini_path['c', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     gini_path['n', gini_type, item] = np.zeros_like(tpi_output['K'])
    #     for t in range(T):
    #         gini_path['b', gini_type, item][t] = inequal.gini(tpi_output['b_mat'][t, :, :], weights[gini_type][t, ...])
    #         gini_path['b_at', gini_type, item][t] = inequal.gini(tpi_output['b_aftertax_path'][t, :, :], weights[gini_type][t, ...])
    #         gini_path['y', gini_type, item][t] = inequal.gini(tpi_output['y_path'][t, :, :], weights[gini_type][t, ...])
    #         gini_path['y_at', gini_type, item][t] = inequal.gini(tpi_output['y_aftertax_path'][t, :, :], weights[gini_type][t, ...])
    #         gini_path['c', gini_type, item][t] = inequal.gini(tpi_output['c_path'][t, :, :], weights[gini_type][t, ...])
    #         gini_path['n', gini_type, item][t] = inequal.gini(tpi_output['n_mat'][t, :, :], weights[gini_type][t, ...])
'''
------------------------------------------------------------------------
    Create figures
------------------------------------------------------------------------
'''

## Labor moments, model vs data
cps = labor.get_labor_data()
labor_dist_data = labor.compute_labor_moments(cps,S)
model_labor_moments = (n_base.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)

domain = np.arange(80) + 20
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(domain, model_labor_moments,
         label='Model', color='black', linestyle='--')
plt.plot(domain, labor_dist_data,
         label='Data', color='black', linestyle='-')
plt.legend()
plt.ylabel(r'household labor supply, \ $\bar{l}_{s}$')
plt.xlabel(r'age-$s$')
labor_dist_comparison = os.path.join(
    GRAPH_DIR, "labor_dist_comparison")
plt.savefig(labor_dist_comparison)
plt.close()

## Labor dist from data with extrapolation
slope = (labor_dist_data[S-20] - labor_dist_data[S-20-15]) / (15.)
to_dot = (-1* slope * xrange(15))[::-1] + labor_dist_data[S-20]
plt.clf()
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(starting_age, S, S-starting_age), labor_dist_data[:S-starting_age], color='black', label='Data')
plt.plot(np.linspace(S, ending_age, ending_age-S), labor_dist_data[S-starting_age:], color='black', linestyle='-.', label='Extrapolation')
plt.plot(np.linspace(65, S, 15), to_dot, linestyle='--', color='black')
plt.axvline(x=S, color='black', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'household labor supply, \ ' r"$\bar{l}_s$")
plt.legend()
labor_dist_data_withfit = os.path.join(
    GRAPH_DIR, "labor_dist_data_withfit")
plt.savefig(labor_dist_data_withfit)
plt.close()

## Calibrated values of chi_n
# est_dir = os.path.join(baseline_dir, "Calibration/chi_estimation.pkl")
# est_output = pickle.load(open(est_dir, "rb"))
# chi_n = est_output.x[J:]
chi_n = chi_n_vec
plt.clf()
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(domain, chi_n,linewidth=2, color='blue')
plt.xlabel(r'Age cohort - $s$')
plt.ylabel(r'$\chi^{n}$')
chi_n = os.path.join(GRAPH_DIR, "chi_n")
plt.savefig(chi_n)
plt.close()

## Wealth over the lifecycle, model vs data for 7 percentile groups
whichpercentile = [0, 25, 50, 70, 80, 90, 99, 100]
data = pd.read_table("./ogusa/data/wealth/scf2007to2013_wealth_age_all_percentiles.csv", sep=',', header=0)
wealth_data = np.array(data)[:, 1:-1]
domain = np.linspace(20, 95, 76)

wealth_data_array = np.ones((78, J))
for j in xrange(J):
    wealth = np.ones((78, whichpercentile[j+1]-whichpercentile[j]))
    for i in xrange(whichpercentile[j]+1, whichpercentile[j+1]):
        wealth_data_array[:,j] = np.mean(wealth_data[:,whichpercentile[j]:whichpercentile[j+1]], axis=1)


wealth_data_tograph = wealth_data_array[2:] / 1000000
wealth_model_tograph = factor_base * bssmat_base[:76] / 1000000


for j in xrange(1,J+1) :
    plt.figure()
    plt.plot(domain, wealth_data_tograph[:, j-1], label='Data')
    plt.plot(domain, wealth_model_tograph[:, j-1], label='Model', linestyle='--')
    plt.xlabel(r'age-$s$')
    plt.ylabel(r'Household savings, in millions of dollars')
    plt.legend(loc=0)
    fig_j = os.path.join(
        GRAPH_DIR, "wealth_fit_graph_{}".format(whichpercentile[j]))
    plt.savefig(fig_j)
    plt.close()

## SS distribution of labor supply, baseline, by ability type
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, n_base[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, n_base[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, n_base[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, n_base[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, n_base[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, n_base[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, n_base[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household labor supply, \ $\bar{l}_{j,s}$')
labor_dist_2D = os.path.join(GRAPH_DIR, "labor_dist_2D")
plt.savefig(labor_dist_2D)
plt.close()

## SS distribution of consumption, baseline, by ability type
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, c_base[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, c_base[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, c_base[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, c_base[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, c_base[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, c_base[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, c_base[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household consumption, \ $\bar{c}_{j,s}$')
consumption_2D = os.path.join(GRAPH_DIR, "consumption_2D")
plt.savefig(consumption_2D)
plt.close()


## SS savings for highest ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, bssmat_base[:, -1], label='Baseline', linestyle='-')
ax.plot(domain, bssmat['income'][:, -1], label='Income Tax Reform', linestyle='--')
ax.plot(domain, bssmat['wealth'][:, -1], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household savings, \ $\bar{b}_{7,s+1}$')
life_save = os.path.join(GRAPH_DIR, "lifecycle_savings_highest_ability")
plt.savefig(life_save)
plt.close()


## SS savings for middle ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, bssmat_base[:, 3], label='Baseline', linestyle='-')
ax.plot(domain, bssmat['income'][:, 3], label='Income Tax Reform', linestyle='--')
ax.plot(domain, bssmat['wealth'][:, 3], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household savings, \ $\bar{b}_{4,s+1}$')
life_save = os.path.join(GRAPH_DIR, "lifecycle_savings_middle_ability")
plt.savefig(life_save)
plt.close()

## SS labor supply for highest ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, n_base[:, -1], label='Baseline', linestyle='-')
ax.plot(domain, n['income'][:, -1], label='Income Tax Reform', linestyle='--')
ax.plot(domain, n['wealth'][:, -1], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household labor supply, \ $\bar{n}_{7,s}$')
life_labor = os.path.join(GRAPH_DIR, "lifecycle_labor_highest_ability")
plt.savefig(life_labor)
plt.close()


## SS labor supply for middle ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, n_base[:, 3], label='Baseline', linestyle='-')
ax.plot(domain, n['income'][:, 3], label='Income Tax Reform', linestyle='--')
ax.plot(domain, n['wealth'][:, 3], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household labor supply, \ $\bar{b}_{4,s}$')
life_labor = os.path.join(GRAPH_DIR, "lifecycle_labor_middle_ability")
plt.savefig(life_labor)
plt.close()


## SS after-tax income for highest ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, income_at_base[:, -1], label='Baseline', linestyle='-')
ax.plot(domain, income_at['income'][:, -1], label='Income Tax Reform', linestyle='--')
ax.plot(domain, income_at['wealth'][:, -1], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household after-tax income, \ $\bar{y}_{7,s}$')
life_income = os.path.join(GRAPH_DIR, "lifecycle_income_highest_ability")
plt.savefig(life_income)
plt.close()


## SS after-tax income for middle ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, income_at_base[:, 3], label='Baseline', linestyle='-')
ax.plot(domain, income_at['income'][:, 3], label='Income Tax Reform', linestyle='--')
ax.plot(domain, income_at['wealth'][:, 3], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household after-tax income, \ $\bar{y}_{4,s}$')
life_income = os.path.join(GRAPH_DIR, "lifecycle_income_middle_ability")
plt.savefig(life_income)
plt.close()


## SS consumption for highest ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, c_base[:, -1], label='Baseline', linestyle='-')
ax.plot(domain, c['income'][:, -1], label='Income Tax Reform', linestyle='--')
ax.plot(domain, c['wealth'][:, -1], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household consumption, \ $\bar{c}_{7,s}$')
life_cons = os.path.join(GRAPH_DIR, "lifecycle_cons_highest_ability")
plt.savefig(life_cons)
plt.close()


## SS consumption for middle ability type; baseline and both reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.plot(domain, c_base[:, 3], label='Baseline', linestyle='-')
ax.plot(domain, c['income'][:, 3], label='Income Tax Reform', linestyle='--')
ax.plot(domain, c['wealth'][:, 3], label='Wealth Tax Reform', linestyle='-.')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'household consumption, \ $\bar{c}_{4,s}$')
life_cons = os.path.join(GRAPH_DIR, "lifecycle_cons_middle_ability")
plt.savefig(life_cons)
plt.close()

## percentage change in consumption over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type

# Compute percent diffs
pct_diff = {}
for item in ('wealth','income'):
    pct_diff['bssmat_'+item] = (bssmat_base - bssmat[item]) / bssmat_base
    pct_diff['n_'+item] = (bssmat_base - n[item]) / bssmat_base
    pct_diff['c_'+item] = (bssmat_base - c[item]) / bssmat_base
    pct_diff['BQ_'+item] = (bssmat_base - BQ[item]) / bssmat_base

for item in ('wealth','income'):
    plt.clf()
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(312)
    # ax.plot(domain, pct_diff['c_'+item][:, 0],
    #         label='0 - 24%', linestyle='-', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 1],
    #         label='25 - 49%', linestyle='--', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 2],
    #         label='50 - 69%', linestyle='-.', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 3],
    #         label='70 - 79%', linestyle=':', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 4],
    #         label='80 - 89%', marker='x', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 5],
    #         label='90 - 99%', marker='v', color='black')
    # ax.plot(domain, pct_diff['c_'+item][:, 6],
    #         label='99 - 100%', marker='1', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 0],
            label='0 - 24%', linestyle='-')
    ax.plot(domain, pct_diff['c_'+item][:, 1],
            label='25 - 49%', linestyle='--')
    ax.plot(domain, pct_diff['c_'+item][:, 2],
            label='50 - 69%', linestyle='-.')
    ax.plot(domain, pct_diff['c_'+item][:, 3],
            label='70 - 79%', linestyle=':')
    ax.plot(domain, pct_diff['c_'+item][:, 4],
            label='80 - 89%', linestyle='-')
    ax.plot(domain, pct_diff['c_'+item][:, 5],
            label='90 - 99%', linestyle='--')
    ax.plot(domain, pct_diff['c_'+item][:, 6],
            label='99 - 100%', linestyle='-.')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .4, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'\% change in \ $\bar{c}_{j,s}$')
    plot_dir = os.path.join(GRAPH_DIR, 'SS_pct_change_c_'+item)
    plt.savefig(plot_dir)
    plt.close()


## percentage change in savings over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type
for item in ('wealth','income'):
    plt.clf()
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(312)
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 0],
    #         label='0 - 24%', linestyle='-', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 1],
    #         label='25 - 49%', linestyle='--', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 2],
    #         label='50 - 69%', linestyle='-.', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 3],
    #         label='70 - 79%', linestyle=':', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 4],
    #         label='80 - 89%', marker='x', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 5],
    #         label='90 - 99%', marker='v', color='black')
    # ax.plot(domain, pct_diff['bssmat_'+item][:, 6],
    #         label='99 - 100%', marker='1', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 0],
            label='0 - 24\%', linestyle='-')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 1],
            label='25 - 49\%', linestyle='--')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 2],
            label='50 - 69\%', linestyle='-.')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 3],
            label='70 - 79\%', linestyle=':')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 4],
            label='80 - 89\%', linestyle='-')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 5],
            label='90 - 99\%', linestyle='--')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 6],
            label='99 - 100\%', linestyle='-.')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .4, box.height])
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'\% change in \ $\bar{b}_{j,s}$')
    plot_dir = os.path.join(GRAPH_DIR, 'SS_pct_change_b_'+item)
    handles, labels = ax.get_legend_handles_labels()
    plt.savefig(plot_dir)
    plt.close()
    # plotting legend separately
    fig_legend = plt.figure(figsize=(3,2), frameon=False)
    axi = fig_legend.add_subplot(111)
    fig_legend.legend(handles, labels, loc='center', ncol=2)
    axi.xaxis.set_visible(False)
    axi.yaxis.set_visible(False)
    for v in axi.spines.values():
        v.set_visible(False)
    axi.axis('off')
    fig_legend.show()
    legend_dir = os.path.join(GRAPH_DIR, 'lifecycle_legend')
    plt.savefig(legend_dir)
    plt.close()

## percentage change in labor supply over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type
for item in ('wealth', 'income'):
    plt.clf()
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(312)
    # ax.plot(domain, pct_diff['n_'+item][:, 0],
    #         label='0 - 24%', linestyle='-', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 1],
    #         label='25 - 49%', linestyle='--', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 2],
    #         label='50 - 69%', linestyle='-.', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 3],
    #         label='70 - 79%', linestyle=':', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 4],
    #         label='80 - 89%', marker='x', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 5],
    #         label='90 - 99%', marker='v', color='black')
    # ax.plot(domain, pct_diff['n_'+item][:, 6],
    #         label='99 - 100%', marker='1', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 0],
            label='0 - 24%', linestyle='-')
    ax.plot(domain, pct_diff['n_'+item][:, 1],
            label='25 - 49%', linestyle='--')
    ax.plot(domain, pct_diff['n_'+item][:, 2],
            label='50 - 69%', linestyle='-.')
    ax.plot(domain, pct_diff['n_'+item][:, 3],
            label='70 - 79%', linestyle=':')
    ax.plot(domain, pct_diff['n_'+item][:, 4],
            label='80 - 89%', linestyle='-')
    ax.plot(domain, pct_diff['n_'+item][:, 5],
            label='90 - 99%', linestyle='--')
    ax.plot(domain, pct_diff['n_'+item][:, 6],
            label='99 - 100%', linestyle='-.')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .4, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'\% change in \ $\bar{n}_{j,s}$')
    plot_dir = os.path.join(GRAPH_DIR, 'SS_pct_change_n_'+item)
    plt.savefig(plot_dir)
    plt.close()

## Mortality rates by age
## Fertility rates by age
## Immigration rates by age - before and after SS change
## Time path of population growth rate
## Initial population distribution by age
## SS population distribution
# handle all of these in demographics.py
# omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector, imm_rates, omega_S_preTP = demographics.get_pop_objs(
#     E, S, T, 1, 100, 2016, True)


## Time path for r in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=rss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{r}$", ls='--')
ax.plot(np.arange(
     T+10), rpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), rpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), rpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Real Interest Rate, \ $\hat{r}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_r')
plt.savefig(plot_dir)
plt.close()

## Time path for w in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=wss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{w}$", ls='--')
ax.plot(np.arange(
     T+10), wpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), wpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), wpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Wage Rate, \ $\hat{w}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_w')
plt.savefig(plot_dir)
plt.close()

# ## Time path for K in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=Kss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{K}$", ls='--')
ax.plot(np.arange(
     T+10), Kpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), Kpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), Kpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Aggregate Capital, \ $\hat{K}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_K')
plt.savefig(plot_dir)
plt.close()

## Time path for L in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=Lss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{L}$", ls='--')
ax.plot(np.arange(
     T+10), Lpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), Lpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), Lpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Aggregate Labor, \ $\hat{L}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_L')
plt.savefig(plot_dir)
plt.close()

# ## Time path for C in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=Css_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{C}$", ls='--')
ax.plot(np.arange(
     T), Cpath['base'][:T], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T), Cpath['income'][:T], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T), Cpath['wealth'][:T], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Aggregate Consumption, \ $\hat{C}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_C')
plt.savefig(plot_dir)
plt.close()


# ## Time path for I in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=Iss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{K}$", ls='--')
ax.plot(np.arange(
     T), Ipath['base'][:T], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T), Ipath['income'][:T], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T), Ipath['wealth'][:T], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Aggregate Investment, \ $\hat{I}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_I')
plt.savefig(plot_dir)
plt.close()


# ## Time path for G in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=Gss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{K}$", ls='--')
ax.plot(np.arange(
     T+10), Gpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), Gpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), Gpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Government Spending, \ $\hat{G}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_G')
plt.savefig(plot_dir)
plt.close()

# ## Time path for T_H in baseline and reforms
plt.clf()
plt.figure()
domain = np.arange(80) + 20
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = plt.subplot(111)
ax.axhline(
    y=T_Hss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{K}$", ls='--')
ax.plot(np.arange(
     T+10), T_Hpath['base'][:T+10], linewidth=2, linestyle='-', label=r"Baseline")
ax.plot(np.arange(
     T+10), T_Hpath['income'][:T+10], linewidth=2, linestyle='--', label=r"Income Tax Reform")
# ax.plot(np.arange(
#      T+10), T_Hpath['wealth'][:T+10], linewidth=2, linestyle='-.', label=r"Wealth Tax Reform")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Year-$t$')
ax.set_ylabel(r'Transfers, \ $\hat{K}_t$')
plot_dir = os.path.join(GRAPH_DIR, 'TPI_T_H')
plt.savefig(plot_dir)
plt.close()


# ## Time path for gini coefficients
gini_name = {'Total': 'total', 'Ability $j$': 'ability', 'Age $s$': 'age'}
for gini_type in ('Total', 'Ability $j$', 'Age $s$'):
    for var_name in ('b', 'b_at', 'y', 'y_at', 'c', 'n'):
        plt.clf()
        plt.figure()
        domain = np.arange(80) + 20
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = plt.subplot(111)
        # ax.axhline(
        #     y=T_Hss_base, color='k', linewidth=2, label=r"Baseline Steady State $\hat{K}$", ls='--')
        ax.plot(np.arange(T), gini_path[var_name, gini_type, 'base'][:T],
                linewidth=2, linestyle='-', label=r"Baseline")
        ax.plot(np.arange(T), gini_path[var_name, gini_type, 'income'][:T],
                linewidth=2, linestyle='--', label=r"Income Tax Reform")
        # ax.plot(np.arange(T), gini_path[var_name, gini_type, 'wealth'][:T],
        #         linewidth=2, linestyle='--', label=r"Wealth Tax Reform")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'Year-$t$')
        ax.set_ylabel(r'Gini Coefficient')
        file_name = 'TPI_gini_' + var_name + '_' + gini_name[gini_type]
        plot_dir = os.path.join(GRAPH_DIR, file_name)
        plt.savefig(plot_dir)
        plt.close()


## compare standard utility to elliptical
n_grid = np.linspace(0.01, 0.8, num=101)
CFE_MU = (1.0/l_tilde)*((n_grid/l_tilde)**theta)
ellipse_MU = (1.0*b_ellipse * (1.0 / l_tilde) * ((1.0 - (n_grid / l_tilde) ** upsilon)
             ** ((1.0 / upsilon) - 1.0)) * (n_grid / l_tilde) ** (upsilon - 1.0))
fig, ax = plt.subplots()
plt.plot(n_grid, CFE_MU, 'r--', label='CFE')
plt.plot(n_grid, ellipse_MU, 'b', label='Elliptical U')
# for the minor ticks, use no labels; default NullFormatter
# ax.xaxis.set_minor_locator(MinorLocator)
# plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.legend(loc='center right')
plt.xlabel(r'Labor Supply')
plt.ylabel(r'Marginal Utility')
plot_dir = os.path.join(GRAPH_DIR, 'EllipseUtilComp')
plt.savefig(plot_dir)
plt.close()


'''
Figres in paper, but that don't need updating:
1) Exogenous income profiles
2) Calibrated wealth tax
3) Calibrated income tax, log
4) Calibrated income tax, not log
5) Model timing
6) Ellipse picture
'''
