'''
------------------------------------------------------------------------
Last updated 8/1/2017

This file sets parameters for the model run.

This py-file calls the following other file(s):
            income.py
            demographics.py
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
Import Packages
------------------------------------------------------------------------
'''
import os
import json
import numpy as np
import scipy.ndimage.filters as filter
from demographics import get_pop_objs
from income import get_e
import pickle
import elliptical_u_est
import matplotlib.pyplot as plt

'''
------------------------------------------------------------------------
Parameters
------------------------------------------------------------------------
Model Parameters:
------------------------------------------------------------------------
S            = integer, number of economically active periods an individual lives
J            = integer, number of different ability groups
T            = integer, number of time periods until steady state is reached
BW           = integer, number of time periods in the budget window
lambdas      = [J,] vector, percentiles for ability groups
imm_rates    = [J,T+S] array, immigration rates by age and year
starting_age = integer, age agents enter population
ending age   = integer, maximum age agents can live until
E            = integer, age agents become economically active
beta_annual  = scalar, discount factor as an annual rate
beta         = scalar, discount factor for model period
sigma        = scalar, coefficient of relative risk aversion
alpha        = scalar, capital share of income
Z            = scalar, total factor productivity parameter in firms' production
               function
delta_annual = scalar, depreciation rate as an annual rate
delta        = scalar, depreciation rate for model period
ltilde       = scalar, measure of time each individual is endowed with each
               period
g_y_annual   = scalar, annual growth rate of technology
g_y          = scalar, growth rate of technology for a model period
frisch       = scalar, Frisch elasticity that is used to fit ellipitcal utility
               to constant Frisch elasticity function
b_ellipse    = scalar, value of b for elliptical fit of utility function
k_ellipse    = scalar, value of k for elliptical fit of utility function
upsilon      = scalar, value of omega for elliptical fit of utility function
------------------------------------------------------------------------
Tax Parameters:
------------------------------------------------------------------------
mean_income_data = scalar, mean income from IRS data file used to calibrate income tax
etr_params       = [S,BW,#tax params] array, parameters for effective tax rate function
mtrx_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                    labor income function
mtry_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                    capital income function
h_wealth         = scalar, wealth tax parameter h (scalar)
m_wealth         = scalar, wealth tax parameter m (scalar)
p_wealth         = scalar, wealth tax parameter p (scalar)
tau_bq           = [J,] vector, bequest tax
tau_payroll      = scalar, payroll tax rate
retire           = integer, age at which individuals eligible for retirement benefits
------------------------------------------------------------------------
Simulation Parameters:
------------------------------------------------------------------------
MINIMIZER_TOL = scalar, tolerance level for the minimizer in the calibration of chi parameters
MINIMIZER_OPTIONS = dictionary, dictionary for options to put into the minimizer, usually
                    to set a max iteration
PLOT_TPI     = boolean, =Ture if plot the path of K as TPI iterates (for debugging purposes)
maxiter      = integer, maximum number of iterations that SS and TPI solution methods will undergo
mindist_SS   = scalar, tolerance for SS solution
mindist_TPI  = scalar, tolerance for TPI solution
nu           = scalar, contraction parameter in SS and TPI iteration process
               representing the weight on the new distribution
flag_graphs  = boolean, =True if produce graphs in demographic, income,
               wealth, and labor files (True=graph)
chi_b_guess  = [J,] vector, initial guess of \chi^{b}_{j} parameters
               (if no calibration occurs, these are the values that will be used for \chi^{b}_{j})
chi_n_guess  = [S,] vector, initial guess of \chi^{n}_{s} parameters
               (if no calibration occurs, these are the values that will be used for \chi^{n}_{s})
------------------------------------------------------------------------
Demographics and Ability variables:
------------------------------------------------------------------------
omega        = [T+S,S] array, time path of stationary distribution of economically active population by age
g_n_ss       = scalar, steady state population growth rate
omega_SS     = [S,] vector, stationary steady state population distribution
surv_rate    = [S,] vector, survival rates by age
rho          = [S,] vector, mortality rates by age
g_n_vector   = [T+S,] vector, growth rate in economically active pop for each period in transition path
e            = [S,J] array, normalized effective labor units by age and ability type
------------------------------------------------------------------------
'''
def get_parameters(baseline, reform, guid, user_modifiable):
    '''
    --------------------------------------------------------------------
    This function sets the parameters for the full model.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if reform
    guid            = string, id for reform run
    user_modifiable = boolean, =True if allow user modifiable parameters
    metadata        = boolean, =True if use metadata file for parameter
                       values (rather than what is entered in parameters below)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    read_tax_func_estimate()
    elliptical_u_est.estimation()
    read_parameter_metadata()

    OBJECTS CREATED WITHIN FUNCTION:
    See parameters defined above
    allvars = dictionary, dictionary with all parameters defined in this function

    RETURNS: allvars

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Model Parameters
    S = int(80) #S<30 won't meet necessary tolerances
    J = int(7)
    T = int(3 * S)
    BW = int(10)
    lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
    #lambdas = np.array([0.5, 0.5])
    #lambdas = np.array([1.,])

    starting_age = 20
    ending_age = 100
    E = int(starting_age * (S / float(ending_age - starting_age)))
    beta_annual = .96 # Carroll (JME, 2009)
    beta = beta_annual ** (float(ending_age - starting_age) / S)
    sigma = 3.0
    alpha = .35 # many use 0.33, but many find that capitals share is
                # increasing (e.g. Elsby, Hobijn, and Sahin (BPEA, 2013))
    Z = 1.0
    delta_annual = .05 # approximately the value from Kehoe calibration
                       # exercise: http://www.econ.umn.edu/~tkehoe/classes/calibration-04.pdf
    delta = 1 - ((1 - delta_annual) ** (float(ending_age - starting_age) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual)**(float(ending_age - starting_age) / S) - 1
    #   Ellipse parameters
    frisch = (1/1.5) # Frisch elasticity consistent with Altonji (JPE, 1996)
                     # and Peterman (Econ Inquiry, 2016)
    b_ellipse, upsilon = elliptical_u_est.estimation(frisch,ltilde)
    k_ellipse = 0 # this parameter is just a level shifter in utlitiy - irrelevant for analysis

    # Tax parameters:
    mean_income_data = 84377.0

    etr_params = np.zeros((S,BW,10))
    mtrx_params = np.zeros((S,BW,10))
    mtry_params = np.zeros((S,BW,10))

    #baseline values - reform values determined in execute.py
    a_tax_income = 3.03452713268985e-06
    b_tax_income = .222
    c_tax_income = 133261.0
    d_tax_income = .219

    etr_params[:,:,0] = a_tax_income
    etr_params[:,:,1] = b_tax_income
    etr_params[:,:,2] = c_tax_income
    etr_params[:,:,3] = d_tax_income

    mtrx_params = etr_params
    mtry_params = etr_params


    #   Wealth tax params
    #       These are non-calibrated values, h and m just need
    #       need to be nonzero to avoid errors. When p_wealth
    #       is zero, there is no wealth tax.
    if reform == 2:
        # wealth tax reform values
        p_wealth = 0.025
        h_wealth = 0.305509008443123
        m_wealth = 2.16050687852062
    else:
        #baseline values
        h_wealth = 0.1
        m_wealth = 1.0
        p_wealth = 0.0



    #   Bequest and Payroll Taxes
    tau_bq = np.zeros(J)
    tau_payroll = 0.15
    retire = np.round(9.0 * S / 16.0) - 1

    # Simulation Parameters
    MINIMIZER_TOL = 1e-14
    MINIMIZER_OPTIONS = None
    PLOT_TPI = False
    maxiter = 250
    mindist_SS = 1e-9
    mindist_TPI = 1e-9 #2e-5
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    # chi_b_guess = np.ones((J,)) * 80.0
    # chi_b_guess = np.array([7.84003265, 10.72762998, 129.97045975, 128.33552107,
    #     229.59424786, 282.90123012, 116.0779987])
    # chi_b_guess = np.array([7.84003265, 10.72762998, 128., 129.,
    #     140., 150., 180.])
    #chi_b_guess = np.array([0.7, 0.7, 1.0, 1.2, 1.2, 1.2, 1.4])
    #chi_b_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 10.0])
    chi_b_guess = np.array([5, 10, 90, 250, 250, 250, 250])
    #chi_b_guess = np.array([2, 10, 90, 350, 1700, 22000, 120000])
    chi_n_guess_80 = np.array([38.12000874, 33.22762421, 25.34842241, 26.67954008, 24.41097278,
                            23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263,
                            21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942,
                            20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951,
                            20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057,
                            19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606,
                            19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741,
                            21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398,
                            22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164,
                            29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997,
                            38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562,
                            37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405,
                            37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256,
                            43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704,
                            53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845,
                            68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.62308152])

    chi_n_guess = filter.uniform_filter(chi_n_guess_80,size=int(80/S))[::int(80/S)]


   # Generate Income and Demographic parameters
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector, imm_rates, omega_S_preTP = get_pop_objs(
        E, S, T, 1, 100, 2016, flag_graphs)

    e = get_e(80, 7, 20, 100, np.array([.25, .25, .2, .1, .1, .09, .01]), flag_graphs)
    # # need to turn 80x7 array into SxJ array
    e /= (e * omega_SS.reshape(S, 1)
                * lambdas.reshape(1, J)).sum()


    allvars = dict(locals())

    return allvars
