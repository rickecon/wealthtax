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
import utils
import os
from scipy import stats
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



'''
------------------------------------------------------------------------
    Create figures
------------------------------------------------------------------------
'''

## Labor moments, model vs data
cps = labor.get_labor_data()
labor_dist_data = labor.compute_labor_moments(cps,S)

plt.figure()
plt.plot(np.arange(80) + 20, (nssmat * lambdas).sum(1),
         label='Model', color='black', linestyle='--')
plt.plot(np.arange(80) + 20, labor_dist_data,
         label='Data', color='black', linestyle='-')
plt.legend()
plt.ylabel(r'individual labor supply $\bar{l}_{s}$')
plt.xlabel(r'age-$s$')
labor_dist_comparison = os.path.join(
    SS_FIG_DIR, "SS/labor_dist_comparison")
plt.savefig(labor_dist_comparison)

## Labor dist from data with extrapolation
plt.plot(domain, weighted, color='black', label='Data')
plt.plot(np.linspace(76, 100, 23), extension, color='black', linestyle='-.', label='Extrapolation')
plt.plot(np.linspace(65, 76, 11), to_dot, linestyle='--', color='black')
plt.axvline(x=76, color='black', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'individual labor supply $/bar{l}_s$')
plt.legend()
plt.savefig('OUTPUT/Demographics/labor_dist_data_withfit.png')

## Calibrated values of chi_n
plt.figure()
plt.plot(domain, chi_n)
plt.xlabel(r'Age cohort - $s$')
plt.ylabel(r'$\chi _n$')
chi_n = os.path.join(SS_FIG_DIR, "SS/chi_n")
plt.savefig(chi_n)

## Wealth over the lifecycle, model vs data for 7 percentile groups

## SS distribution of labor supply, baseline, by ability type
fig113 = plt.figure()
ax = plt.subplot(111)
ax.plot(domain, nssmat[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, nssmat[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, nssmat[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, nssmat[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, nssmat[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, nssmat[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, nssmat[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'individual labor supply $\bar{l}_{j,s}$')
labor_dist_2D = os.path.join(SS_FIG_DIR, "SS/labor_dist_2D")
plt.savefig(labor_dist_2D)

## SS distribution of consumption, baseline, by ability type
fig114 = plt.figure()
ax = plt.subplot(111)
ax.plot(domain, cssmat[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, cssmat[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, cssmat[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, cssmat[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, cssmat[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, cssmat[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, cssmat[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'individual consumption $\bar{c}_{j,s}$')
consumption_2D = os.path.join(SS_FIG_DIR, "SS/consumption_2D")
plt.savefig(consumption_2D)

## percentage change in consumption over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type

## percentage change in savings over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type

## percentage change in labor supply over lifecycle, baseline vs reform (wealth
## and income tax reforms), separate for each type

## Mortality rates by age

## Fertility rates by age

## Immigration rates by age - before and after SS change

## Time path of population growth rate

## Initial population distribution by age

## SS population distribution

## Time path for K

## Time path for L

## compare standard utility to elliptical


'''
Figres in paper, but that don't need updating:
1) Exogenous income profiles
2) Calibrated wealth tax
3) Calibrated income tax, log
4) Calibrated income tax, not log
5) Model timing
6) Ellipse picture
'''
