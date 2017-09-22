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
#income_base = ss_output[]
Kss_base = ss_output['Kss']
Lss_base = ss_output['Lss']


bssmat = {}
factor = {}
n = {}
c = {}
BQ = {}
for item in ('wealth','income'):
    ss_dir = os.path.join(reform_dir[item], "sigma2.0/SS/SS_vars.pkl")
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
plt.ylabel(r'individual labor supply, \ $\bar{l}_{s}$')
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
plt.ylabel(r'individual labor supply, \ ' r"$\bar{l}_s$")
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
    plt.ylabel(r'Individual savings, in millions of dollars')
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
ax.set_ylabel(r'individual labor supply, \ $\bar{l}_{j,s}$')
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
ax.set_ylabel(r'individual consumption, \ $\bar{c}_{j,s}$')
consumption_2D = os.path.join(GRAPH_DIR, "consumption_2D")
plt.savefig(consumption_2D)
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
    ax.plot(domain, pct_diff['c_'+item][:, 0],
            label='0 - 24%', linestyle='-', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 1],
            label='25 - 49%', linestyle='--', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 2],
            label='50 - 69%', linestyle='-.', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 3],
            label='70 - 79%', linestyle=':', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 4],
            label='80 - 89%', marker='x', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 5],
            label='90 - 99%', marker='v', color='black')
    ax.plot(domain, pct_diff['c_'+item][:, 6],
            label='99 - 100%', marker='1', color='black')
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
    ax.plot(domain, pct_diff['bssmat_'+item][:, 0],
            label='0 - 24%', linestyle='-', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 1],
            label='25 - 49%', linestyle='--', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 2],
            label='50 - 69%', linestyle='-.', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 3],
            label='70 - 79%', linestyle=':', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 4],
            label='80 - 89%', marker='x', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 5],
            label='90 - 99%', marker='v', color='black')
    ax.plot(domain, pct_diff['bssmat_'+item][:, 6],
            label='99 - 100%', marker='1', color='black')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .4, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'\% change in \ $\bar{b}_{j,s}$')
    plot_dir = os.path.join(GRAPH_DIR, 'SS_pct_change_b_'+item)
    plt.savefig(plot_dir)
    plt.close()

## percentage change in labor supply over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type
for item in ('wealth','income'):
    plt.clf()
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(312)
    ax.plot(domain, pct_diff['n_'+item][:, 0],
            label='0 - 24%', linestyle='-', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 1],
            label='25 - 49%', linestyle='--', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 2],
            label='50 - 69%', linestyle='-.', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 3],
            label='70 - 79%', linestyle=':', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 4],
            label='80 - 89%', marker='x', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 5],
            label='90 - 99%', marker='v', color='black')
    ax.plot(domain, pct_diff['n_'+item][:, 6],
            label='99 - 100%', marker='1', color='black')
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


# ## Time path for K in baseline
tpi_dir = os.path.join(baseline_dir, "sigma2.0/TPI/TPI_vars.pkl")
tpi_output = pickle.load(open(tpi_dir, "rb"))
Kpath_TPI = tpi_output['K']
plt.figure()
plt.axhline(
    y=Kss_base, color='r', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
plt.plot(np.arange(
     T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
plot_dir = os.path.join(GRAPH_DIR, 'TPI_K')
plt.savefig(plot_dir)
plt.close()

## Time path for L
Lpath_TPI = tpi_output['L']
plt.figure()
plt.axhline(
    y=Lss_base, color='r', linewidth=2, label=r"Steady State $\hat{L}$", ls='--')
plt.plot(np.arange(
     T+10), Lpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{L}_t$")
plot_dir = os.path.join(GRAPH_DIR, 'TPI_L')
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
