'''
------------------------------------------------------------------------
Last updated: 4/8/2016

Calculates steady state of OG-USA model with S age cohorts and J
ability types.

This py-file calls the following other file(s):
            tax.py
            household.py
            firm.py
            utils.py
            OUTPUT/SS/ss_vars.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/SS/ss_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import scipy.optimize as opt
import cPickle as pickle

from . import tax
from . import household
import firm
import utils
import os


'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = False

'''
Grab some values from prior run to serve as starting values
'''
#START_VALUES = pickle.load(open("./ogusa/SS_vars_sigma2.0_baseline.pkl", "rb"))
#START_VALUES = pickle.load(open("./ogusa/SS_vars_sigma2.0_wealth.pkl", "rb"))
#START_VALUES = pickle.load(open("./ogusa/SS_vars_sigma3.0_baseline.pkl", "rb"))
#START_VALUES = pickle.load(open("./OUTPUT_INCOME_REFORM/sigma2.0/SS/SS_vars.pkl", "rb"))
#START_VALUES = pickle.load(open("./OUTPUT_WEALTH_REFORM/sigma3.0/SS/SS_vars.pkl", "rb"))
# START_VALUES = pickle.load(open("./OUTPUT_BASELINE/SS/SS_vars.pkl", "rb"))
START_VALUES = pickle.load(open("./OUTPUT_BASELINE/sigma2.0/SS/SS_vars.pkl", "rb"))


'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''

def create_steady_state_parameters(**sim_params):
    '''
    --------------------------------------------------------------------
    This function calls the tax function estimation routine and saves
    the resulting dictionary in pickle files corresponding to the
    baseline or reform policy.
    --------------------------------------------------------------------

    INPUTS:
    sim_params       = dictionary, dict containing variables for simulation
    analytical_mtrs  = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params       = [S,BW,#tax params] array, parameters for effective tax rate function
    mtrx_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                       labor income function
    mtry_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                       capital income function
    b_ellipse        = scalar, value of b for elliptical fit of utility function
    upsilon          = scalar, value of omega for elliptical fit of utility function
    S                = integer, number of economically active periods an individual lives
    J                = integer, number of different ability groups
    T                = integer, number of time periods until steady state is reached
    BW               = integer, number of time periods in the budget window
    beta             = scalar, discount factor for model period
    sigma            = scalar, coefficient of relative risk aversion
    alpha            = scalar, capital share of income
    Z                = scalar, total factor productivity parameter in firms' production
                       function
    ltilde           = scalar, measure of time each individual is endowed with each
                       period
    nu               = scalar, contraction parameter in SS and TPI iteration process
                       representing the weight on the new distribution
    g_y              = scalar, growth rate of technology for a model period
    tau_payroll      = scalar, payroll tax rate
    retire           = integer, age at which individuals eligible for retirement benefits
    mean_income_data = scalar, mean income from IRS data file used to calibrate income tax
    run_params       = ???
    output_dir       = string, directory for output files to be saved


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    income_tax_params = length 3 tuple, (analytical_mtrs, etr_params,
                        mtrx_params,mtry_params)
    wealth_tax_params = [3,] vector, contains values of three parameters
                        of wealth tax function
    ellipse_params    = [2,] vector, vector with b_ellipse and upsilon
                        paramters of elliptical utility
    parameters        = length 3 tuple, ([15,] vector of general model
                        params, wealth_tax_params, ellipse_params)
    iterative_params  = [2,] vector, vector with max iterations and tolerance
                        for SS solution

    RETURNS: (income_tax_params, wealth_tax_params, ellipse_params,
            parameters, iterative_params)

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Put income tax parameters in a tuple
    # Assumption here is that tax parameters of last year of budget
    # window continue forever and so will be SS values
    income_tax_params = (sim_params['analytical_mtrs'], sim_params['etr_params'][:,-1,:],
                         sim_params['mtrx_params'][:,-1,:],sim_params['mtry_params'][:,-1,:])

    # Make a vector of all one dimensional parameters, to be used in the
    # following functions
    wealth_tax_params = [sim_params['h_wealth'], sim_params['p_wealth'], sim_params['m_wealth']]
    ellipse_params = [sim_params['b_ellipse'], sim_params['upsilon']]

    ss_params = [sim_params['J'], sim_params['S'], sim_params['T'], sim_params['BW'],
                  sim_params['beta'], sim_params['sigma'], sim_params['alpha'],
                  sim_params['Z'], sim_params['delta'], sim_params['ltilde'],
                  sim_params['nu'], sim_params['g_y'], sim_params['g_n_ss'],
                  sim_params['tau_payroll'], sim_params['tau_bq'], sim_params['rho'], sim_params['omega_SS'],
                  sim_params['lambdas'], sim_params['imm_rates'][-1,:], sim_params['e'], sim_params['retire'], sim_params['mean_income_data']] + \
                  wealth_tax_params + ellipse_params
    iterative_params = [sim_params['maxiter'], sim_params['mindist_SS']]
    chi_params = (sim_params['chi_b_guess'], sim_params['chi_n_guess'])
    return (income_tax_params, ss_params, iterative_params, chi_params)

def lifecycle_solver(guess, params):
    '''
    --------------------------------------------------------------------
    Finds b and n over the lifecyle, one ability type at a time.
    --------------------------------------------------------------------

    INPUTS:
    guesses = scalar, initial guesses for b_S+1
    r = scalar, real interest rate
    w = scalar, real wage rate
    T_H = scalar, lump sum transfer
    factor = scalar, scaling factor converting model units to dollars
    j = integer, ability group
    params = length 21 tuple, list of parameters
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    tau_bq = scalar, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega_SS = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type
    tax_params = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params      = [S,BW,#tax params] array, parameters for effective tax rate function
    mtrx_params     = [S,BW,#tax params] array, parameters for marginal tax rate on
                       labor income function
    mtry_params     = [S,BW,#tax params] array, parameters for marginal tax rate on
                       capital income function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    household.get_BQ()
    tax.replacement_rate_vals()
    household.FOC_savings()
    household.FOC_labor()
    tax.total_taxes()
    household.get_cons()

    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor ss
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    b_splus2 = [S,] vector, household savings one period ahead
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    RETURNS: 2Sx1 list of euler errors

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    r, w, T_H, BQ, theta, factor, j, J, S, beta, sigma, ltilde, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon,\
                  j, chi_b, chi_n, tau_bq, rho, lambdas, omega_SS, e,\
                  analytical_mtrs, etr_params, mtrx_params,\
                  mtry_params = params

    # BQ_params = (omega_SS, lambdas[j], rho, g_n_ss, 'SS')
    # BQ = household.get_BQ(r, b_splus1, BQ_params)
    #theta_params = (e[:,j], S, J, omega_SS, lambdas[j],retire)
    # theta_params = (e, S, J, omega_SS, lambdas,retire)
    # theta = tax.replacement_rate_vals(n_guess, w, factor, theta_params)


    # initialize arrays of HH vars
    c = np.zeros((S,))
    n = np.zeros((S,))
    b = np.zeros((S+1,))
    b[-1] = guess

    # use FOC for save in last period of life to solve for c_S
    c[-1] = (chi_b[j]**(-1/sigma))*b[-1]

    # backwards induct over lifetime to find path of other variables
    for s in range(S-1):
        # solve for n and b
        if s == 0:
            guesses = [0.4, b[S-s]]
        else:
            guesses = [n[S-s-1], b[S-s-1]] # guess for n and b

        n_b_params = [b[S-s], r, w, T_H, BQ[j], theta[j], factor, c[S-s-1], e[:, j], sigma, g_y, theta[j],\
                     b_ellipse, upsilon, chi_n, \
                     ltilde, tau_bq[j], lambdas[j], J, S, analytical_mtrs, \
                     etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, \
                     tau_payroll, retire, 'SS', s]
        [solutions, infodict, ier, message] = opt.fsolve(solve_n_b, guesses,
                                    args=n_b_params, xtol=MINIMIZER_TOL, full_output=True)
        # print 'n_b_errors: ', infodict['fvec']
        n[S-s-1] = solutions[0]
        b[S-s-1] = solutions[1]
        c_params = [c[S-s-1], n[S-s-1], b[S-s-1], r, w, T_H, BQ[j], theta[j], factor, e[:,j], sigma, beta, g_y, chi_b[j], tau_bq, rho, lambdas[j], J, S, \
            analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, 'SS', s]
        guess = c[S-s-1]
        [solution, infodict, ier, message] = opt.fsolve(household.solve_c, guess,
                                    args=c_params, xtol=MINIMIZER_TOL, full_output=True)
        # print 'c_errors: ', infodict['fvec']
        c[S-s-2] = solution

    # solve for n in initial year of life
    n_b_params = [b[1], r, w, T_H, BQ[j], theta[j], factor, c[0], e[:, j], sigma, g_y, theta[j],\
                 b_ellipse, upsilon, chi_n, \
                 ltilde, tau_bq[j], lambdas[j], J, S, analytical_mtrs, \
                 etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, \
                 tau_payroll, retire, 'SS', S-1]
    guesses = [n[1], b[1]]
    [solutions, infodict, ier, message] = opt.fsolve(solve_n_b, guesses,
                                args=n_b_params, xtol=MINIMIZER_TOL, full_output=True)
    # print 'n_b_errors: ', infodict['fvec']
    n[0] = solutions[0]
    b[0] = solutions[1]

    return b, n, c

def lc_error(guess, params):
    # check that get back to b[0]=0
    b, n, c = lifecycle_solver(guess, params)
    error = b[0]
    return error


def solve_n_b(guesses, params):
    '''
    --------------------------------------------------------------------
    Uses the FOC for labor and the BC to solve for n and b in the current period
    --------------------------------------------------------------------
    '''
    b_splus1, r, w, T_H, BQ, theta, factor, c, e, sigma, g_y, theta, b_ellipse, upsilon, chi_n, \
                 ltilde, tau_bq, lambdas, J, S, analytical_mtrs, \
                 etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, \
                 tau_payroll, retire, method, s = params

    # also need to have c[s] input to this function
    # need to change so FOC_labor takes c[s],b[s+1] and guess for b,n and outputs error

    n = guesses[0]
    b = guesses[1]

    foc_labor_params = [e, sigma, g_y, theta, b_ellipse, upsilon, chi_n,
                 ltilde, tau_bq, lambdas, J, S, analytical_mtrs,
                 etr_params, mtrx_params, h_wealth, p_wealth, m_wealth,
                 tau_payroll, retire, method]
    error1 = household.FOC_labor(r, w, b, b_splus1, n, BQ, factor, T_H, foc_labor_params)[S-s-1]

    # need to change get_cons so takes c[s],b[s+1] and guess for b,n and outputs error
    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth,
                   m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    error2 = ((1 + r) * b) + (w * e[S-s-1] * n) + (BQ/lambdas) - (b_splus1 * np.exp(g_y)) - tax1[S-s-1] - c

    # print 'n = ', n
    # print 'b = ', b
    # if n <= 0.0:
    #     error1 = 1e14
    # if n >= 1.0:
    #     error1 = 1e14
    # # if b <= 0.0:
    # #     error1 = 1e14
    return [error1,error2]


def inner_loop(outer_loop_vars, params, baseline):
    '''
    This function solves for the inner loop of
    the SS.  That is, given the guesses of the
    outer loop variables (r, w, T_H, factor)
    this function solves the households'
    problems in the SS.

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        T_H        = [T,] vector, lump sum transfer amount(s)


    Functions called:
        euler_equation_solver()
        household.get_K()
        firm.get_L()
        firm.get_Y()
        firm.get_r()
        firm.get_w()
        household.get_BQ()
        tax.replacement_rate_vals()
        tax.get_lump_sum()

    Objects in function:


    Returns: euler_errors, bssmat, nssmat, new_r, new_w
             new_T_H, new_factor, new_BQ

    '''

    # unpack variables and parameters pass to function
    bssmat, nssmat, r, w, T_H, BQ, theta, factor = outer_loop_vars
    ss_params, income_tax_params, chi_params = params

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    chi_b, chi_n = chi_params

    # bssmat = START_VALUES['bssmat_splus1']
    # nssmat = START_VALUES['nssmat']
    cssmat = np.zeros((S,J))
    euler_errors = np.zeros((2*S,J))

    for j in xrange(J):
        # Solve the euler equations
        if j == 0:
            b_Sp1_guess = bssmat[-1, j]
        else:
            b_Sp1_guess = bssmat[-1, j-1]*10

        euler_params = [r, w, T_H, BQ, theta, factor, j, J, S, beta, sigma, ltilde, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon,\
                  j, chi_b, chi_n, tau_bq, rho, lambdas, omega_SS, e,\
                  analytical_mtrs, etr_params, mtrx_params,\
                  mtry_params]

        [solution, infodict, ier, message] = opt.fsolve(lc_error, b_Sp1_guess,
                                     args=euler_params, xtol=MINIMIZER_TOL, full_output=True)
        # [x0, r_out] = opt.bisect(lc_error, -1.0, 10.0, args=euler_params, xtol=MINIMIZER_TOL, full_output=True, disp=False)
        print 'j = ', j
        print 'b[0] error = ', infodict['fvec']
        print 'message: ', message
        # print 'b[S]= ', x0
        # print 'converged= ', r_out.converged

        b_out, nssmat[:, j], cssmat[:, j] = lifecycle_solver(solution,euler_params)
        bssmat[:, j] = b_out[1:]
        # print solutions
        # quit()
        #
        # euler_errors[:,j] = infodict['fvec']
        # print 'j = ', j
        # print 'Max Euler errors: ', np.absolute(euler_errors[:,j]).max()
    # print 'bssmat: ', bssmat
    # print 'nssmat: ', nssmat
    # print 'cssmat: ', cssmat
    quit()
    K_params = (omega_SS.reshape(S, 1), lambdas.reshape(1, J), imm_rates, g_n_ss, 'SS')
    K = household.get_K(bssmat, K_params)
    L_params = (e, omega_SS.reshape(S, 1), lambdas.reshape(1, J), 'SS')
    L = firm.get_L(nssmat, L_params)
    Y_params = (alpha, Z)
    Y = firm.get_Y(K, L, Y_params)
    r_params = (alpha, delta)
    new_r = firm.get_r(Y, K, r_params)
    new_w = firm.get_w(Y, L, alpha)
    b_s = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat[:-1, :]))
    average_income_model = ((new_r * b_s + new_w * e * nssmat) *
                            omega_SS.reshape(S, 1) *
                            lambdas.reshape(1, J)).sum()
    if baseline:
        new_factor = mean_income_data / average_income_model
    else:
        new_factor = factor

    BQ_params = (omega_SS.reshape(S, 1), lambdas.reshape(1, J), rho.reshape(S, 1), g_n_ss, 'SS')
    new_BQ = household.get_BQ(new_r, bssmat, BQ_params)
    theta_params = (e, S, J, omega_SS.reshape(S, 1), lambdas,retire)
    new_theta = tax.replacement_rate_vals(nssmat, new_w, new_factor, theta_params)

    T_H_params = (e, lambdas.reshape(1, J), omega_SS.reshape(S, 1), 'SS', etr_params, theta, tau_bq,
                      tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J)
    new_T_H = tax.get_lump_sum(new_r, new_w, b_s, nssmat, new_BQ, factor, T_H_params)

    print 'Inner Loop Max Euler Error: ', (np.absolute(euler_errors)).max()
    # print 'K: ', K
    # print 'L: ', L
    #print 'bssmat: ', bssmat
    return euler_errors, bssmat, nssmat, new_r, new_w, \
             new_T_H, new_BQ, new_theta, new_factor, average_income_model




def SS_solver(b_guess_init, n_guess_init, wss, rss, T_Hss, BQss, factor_ss, params, baseline, fsolve_flag=False):
    '''
    --------------------------------------------------------------------
    Solves for the steady state distribution of capital, labor, as well as
    w, r, T_H and the scaling factor, using a bisection method similar to TPI.
    --------------------------------------------------------------------

    INPUTS:
    b_guess_init = [S,J] array, initial guesses for savings
    n_guess_init = [S,J] array, initial guesses for labor supply
    wguess = scalar, initial guess for SS real wage rate
    rguess = scalar, initial guess for SS real interest rate
    T_Hguess = scalar, initial guess for lump sum transfer
    factorguess = scalar, initial guess for scaling factor to dollars
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    params = lenght X tuple, list of parameters
    iterative_params = length X tuple, list of parameters that determine the convergence
                       of the while loop
    tau_bq = [J,] vector, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    euler_equation_solver()
    household.get_K()
    firm.get_L()
    firm.get_Y()
    firm.get_r()
    firm.get_w()
    household.get_BQ()
    tax.replacement_rate_vals()
    tax.get_lump_sum()
    utils.convex_combo()
    utils.pct_diff_func()


    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor supply
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    b_splus2 = [S,] vector, household savings one period ahead
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    RETURNS: solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)

    OUTPUT: None
    --------------------------------------------------------------------
    '''

    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params = params

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    # Rename the inputs
    w = wss
    r = rss
    T_H = T_Hss
    BQ = BQss
    factor = factor_ss

    dist = 10
    iteration = 0
    dist_vec = np.zeros(maxiter)

    if fsolve_flag == True:
        maxiter = 1

    while (dist > mindist_SS) and (iteration < maxiter):
        # Solve for the steady state levels of b and n, given w, r, T_H and
        # factor

        outer_loop_vars = (bssmat, nssmat, r, w, T_H, factor)
        inner_loop_params = (ss_params, income_tax_params, chi_params)

        euler_errors, bssmat, nssmat, new_r, new_w, \
             new_T_H, new_BQ, new_theta, new_factor, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline)

        # print 'T_H: ', T_H, new_T_H
        # print 'factor: ', factor, new_factor
        # print 'interest rate: ', r, new_r
        # print 'wage rate: ', w, new_w

        r = utils.convex_combo(new_r, r, nu)
        w = utils.convex_combo(new_w, w, nu)
        factor = utils.convex_combo(new_factor, factor, nu)
        T_H = utils.convex_combo(new_T_H, T_H, nu)
        BQ = utils.convex_combo(new_BQ, BQ, nu)
        theta = utils.convex_combo(new_theta, theta, nu)
        if T_H != 0:
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            [utils.pct_diff_func(new_w, w)] +
                            [utils.pct_diff_func(new_T_H, T_H)] +
                            [utils.pct_diff_func(new_BQ, BQ)] +
                            [utils.pct_diff_func(new_theta, theta)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()
        else:
            # If T_H is zero (if there are no taxes), a percent difference
            # will throw NaN's, so we use an absoluate difference
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            [utils.pct_diff_func(new_w, w)] +
                            [abs(new_T_H - T_H)] +
                            [utils.pct_diff_func(new_BQ, BQ)] +
                            [utils.pct_diff_func(new_theta, theta)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()
        dist_vec[iteration] = dist
        # Similar to TPI: if the distance between iterations increases, then
        # decrease the value of nu to prevent cycling
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration - 1] > 0:
                nu /= 2.0
                #print 'New value of nu:', nu
        iteration += 1
        #print "Iteration: %02d" % iteration, " Distance: ", dist

    '''
    ------------------------------------------------------------------------
        Generate the SS values of variables, including euler errors
    ------------------------------------------------------------------------
    '''
    bssmat_s = np.append(np.zeros((1,J)),bssmat[:-1,:],axis=0)
    bssmat_splus1 = bssmat

    wss = w
    rss = r
    factor_ss = factor
    T_Hss = T_H

    Kss_params = (omega_SS.reshape(S, 1), lambdas, imm_rates, g_n_ss, 'SS')
    Kss = household.get_K(bssmat_splus1, Kss_params)
    Lss_params = (e, omega_SS.reshape(S, 1), lambdas, 'SS')
    Lss = firm.get_L(nssmat, Lss_params)
    Yss_params = (alpha, Z)
    Yss = firm.get_Y(Kss, Lss, Yss_params)
    Iss_params = (delta, g_y, omega_SS, lambdas, imm_rates, g_n_ss, 'SS')
    Iss = firm.get_I(bssmat_splus1, Kss, Kss, Iss_params)

    BQss = new_BQ
    # theta_params = (e, S, J, omega_SS.reshape(S, 1), lambdas,retire)
    # theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, theta_params)

    # solve resource constraint
    etr_params_3D = np.tile(np.reshape(etr_params,(S,1,etr_params.shape[1])),(1,J,1))
    mtrx_params_3D = np.tile(np.reshape(mtrx_params,(S,1,mtrx_params.shape[1])),(1,J,1))
    taxss_params = (e, lambdas, 'SS', retire, etr_params_3D,
                    h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    taxss = tax.total_taxes(rss, wss, bssmat_s, nssmat, BQss, factor_ss, T_Hss, None, False, taxss_params)
    css_params = (e, lambdas.reshape(1, J), g_y)
    cssmat = household.get_cons(rss, wss, bssmat_s, bssmat_splus1, nssmat, BQss.reshape(
        1, J), taxss, css_params)

    Css_params = (omega_SS.reshape(S, 1), lambdas, 'SS')
    Css = household.get_C(cssmat, Css_params)

    resource_constraint = Yss - (Css + Iss)



    '''
    ------------------------------------------------------------------------
        The code below is to calulate and save model MTRs
                - only exists to help debug
    ------------------------------------------------------------------------
    '''
    # etr_params_extended = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
    # etr_params_extended_3D = np.tile(np.reshape(etr_params_extended,(S,1,etr_params_extended.shape[1])),(1,J,1))
    # mtry_params_extended = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]
    # mtry_params_extended_3D = np.tile(np.reshape(mtry_params_extended,(S,1,mtry_params_extended.shape[1])),(1,J,1))
    # e_extended = np.array(list(e) + list(np.zeros(J).reshape(1, J)))
    # nss_extended = np.array(list(nssmat) + list(np.zeros(J).reshape(1, J)))
    # mtry_ss_params = (e_extended[1:,:], etr_params_extended_3D, mtry_params_extended_3D, analytical_mtrs)
    # mtry_ss = tax.MTR_capital(rss, wss, bssmat_splus1, nss_extended[1:,:], factor_ss, mtry_ss_params)
    # mtrx_ss_params = (e, etr_params_3D, mtrx_params_3D, analytical_mtrs)
    # mtrx_ss = tax.MTR_labor(rss, wss, bssmat_s, nssmat, factor_ss, mtrx_ss_params)
    #
    # etr_ss_params = (e, etr_params_3D)
    # etr_ss = tax.tau_income(rss, wss, bssmat_s, nssmat, factor_ss, etr_ss_params)
    #
    # np.savetxt("etr_ss.csv", etr_ss, delimiter=",")
    # np.savetxt("mtr_ss_capital.csv", mtry_ss, delimiter=",")
    # np.savetxt("mtr_ss_labor.csv", mtrx_ss, delimiter=",")


    print 'interest rate: ', rss
    print 'wage rate: ', wss
    print 'factor: ', factor_ss
    print 'T_H', T_Hss
    print 'Resource Constraint Difference:', resource_constraint
    print 'Max Euler Error: ', (np.absolute(euler_errors)).max()

    if ENFORCE_SOLUTION_CHECKS and np.absolute(resource_constraint) > 1e-8:
        err = "Steady state aggregate resource constraint not satisfied"
        raise RuntimeError(err)

    # check constraints
    household.constraint_checker_SS(bssmat, nssmat, cssmat, ltilde)

    if np.absolute(resource_constraint) > 1e-8 or (np.absolute(euler_errors)).max() > 1e-8:
        ss_flag = 1
    else:
        ss_flag = 0


    euler_savings = euler_errors[:S,:]
    euler_labor_leisure = euler_errors[S:,:]

    '''
    ------------------------------------------------------------------------
        Return dictionary of SS results
    ------------------------------------------------------------------------
    '''

    output = {'Kss': Kss, 'bssmat': bssmat, 'Lss': Lss, 'Css':Css, 'Iss':Iss,
              'nssmat': nssmat, 'Yss': Yss,'wss': wss, 'rss': rss, 'theta': theta,
              'BQss': BQss, 'factor_ss': factor_ss, 'bssmat_s': bssmat_s,
              'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'T_Hss': T_Hss, 'euler_savings': euler_savings,
              'euler_labor_leisure': euler_labor_leisure, 'chi_n': chi_n,
              'chi_b': chi_b, 'ss_flag':ss_flag}

    return output



def SS_fsolve(guesses, params):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, T_H and the scaling factor, using an a root finder.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factorguess = guess for scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the convergence
                           of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''

    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params = params

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    baseline = True

    # Rename the inputs
    w = guesses[0]
    r = guesses[1]
    T_H = guesses[2]
    BQ = guesses[2:2+J]
    theta = guesses[2+J:2+2*J]
    factor = guesses[-1]

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    outer_loop_vars = (bssmat, nssmat, r, w, T_H, BQ, theta, factor)
    inner_loop_params = (ss_params, income_tax_params, chi_params)
    euler_errors, bssmat_out, nssmat_out, new_r, new_w, \
         new_T_H, new_BQ, new_theta, new_factor, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline)

    # only update initial guesses of b and n if HH problem solved
    # if (np.absolute(euler_errors)).max() < 1e-08:
    #     bssmat = bssmat_out
    #     nssmat = nssmat_out


    error1 = new_w - w
    error2 = new_r - r
    error3 = new_T_H - T_H
    error_BQ = new_BQ - BQ
    error4 = new_theta - theta
    error5 = new_factor/1000000 - factor/1000000


    # print 'mean income in model and data: ', average_income_model, mean_income_data
    # print 'model income with factor: ', average_income_model*factor
    #
    # print 'errors: ', error1, error2, error3, error4
    print 'T_H: ', T_H, new_T_H
    print 'factor: ', factor, new_factor
    print 'interest rate: ', r, new_r
    print 'wage rate: ', w, new_w
    print 'BQ: ', BQ, new_BQ
    print 'theta: ', theta, new_theta
    quit()

    # Check and punish violations
    if w <= 0:
        error1 = 1e14
    if np.isnan(w):
        error1 = 1e14
    if r <= 0:
        error2 = 1e14
    if np.isnan(r):
        error2 = 1e14
    if r > 1:
        error2 = 1e9
    if T_H <= 0:
        error3 = 1e14
    if np.isnan(T_H):
        error3 = 1e14
    if factor <= 0:
        error5 = 1e14
    if np.isnan(factor):
        error5 = 1e14

    errors = guesses = np.append([error1, error2, error3],np.append([error_BQ.flatten()],np.append([error4.flatten()],[error5])))
    print 'errors: ', errors

    return errors




def SS_fsolve_reform(guesses, params):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, and T_H and the scaling factor, using a root finder. This solves for the
    reform SS and so takes the factor from the baseline SS as an input.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factor = scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the convergence
                           of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''
    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params, factor = params

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    baseline = False

    # Rename the inputs
    w = guesses[0]
    r = guesses[1]
    T_H = guesses[2]

    print 'Reform SS factor is: ', factor

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    outer_loop_vars = (bssmat, nssmat, r, w, T_H, factor)
    inner_loop_params = (ss_params, income_tax_params, chi_params)

    euler_errors, bssmat, nssmat, new_r, new_w, \
        new_T_H, new_factor, new_BQ, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline)

    error1 = new_w - w
    error2 = new_r - r
    error3 = new_T_H - T_H
    print 'errors: ', error1, error2, error3
    print 'T_H: ', new_T_H


    # Check and punish violations
    if r <= 0:
        error1 += 1e9
    #if r > 1:
    #    error1 += 1e9
    if w <= 0:
        error2 += 1e9

    return [error1, error2, error3]



def run_SS(income_tax_params, ss_params, iterative_params, chi_params, baseline=True, baseline_dir="./OUTPUT"):
    '''
    --------------------------------------------------------------------
    Solve for SS of OG-USA.
    --------------------------------------------------------------------

    INPUTS:
    income_tax_parameters = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    ss_parameters = length 21 tuple, (J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params  = [2,] vector, vector with max iterations and tolerance
                        for SS solution
    baseline = boolean, =True if run is for baseline tax policy
    calibrate_model = boolean, =True if run calibration of chi parameters
    output_dir = string, path to save output from current model run
    baseline_dir = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS_fsolve()

    OBJECTS CREATED WITHIN FUNCTION:
    chi_params = [J+S,] vector, chi_b and chi_n stacked together
    b_guess = [S,J] array, initial guess at savings
    n_guess = [S,J] array, initial guess at labor supply
    wguess = scalar, initial guess at SS real wage rate
    rguess = scalar, initial guess at SS real interest rate
    T_Hguess = scalar, initial guess at SS lump sum transfers
    factorguess = scalar, initial guess at SS factor adjustment (to scale model units to dollars)

    output


    RETURNS: output

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    # b_guess = np.ones((S, J)).flatten() * 0.05
    # n_guess = np.ones((S, J)).flatten() * .4 * ltilde
    b_guess = START_VALUES['bssmat_splus1']
    n_guess = START_VALUES['nssmat']
    # For initial guesses of w, r, T_H, and factor, we use values that are close
    # to some steady state values.

    if baseline:
        wguess = START_VALUES['wss'] #0.968167841907 #1.16
        rguess = START_VALUES['rss'] #0.116998690192 #.068
        T_Hguess = START_VALUES['T_Hss'] #0.0304546765599 #0.046
        factorguess = START_VALUES['factor_ss'] #274072.825051 #239344.894517

        BQ_guess = np.zeros((J,))
        for j in range(J):
            BQ_params = (omega_SS, lambdas[j], rho, g_n_ss, 'SS')
            BQ_guess[j] = household.get_BQ(rguess, b_guess[:,j], BQ_params)
        theta_params = (e, S, J, omega_SS, lambdas,retire)
        theta_guess = tax.replacement_rate_vals(n_guess, wguess, factorguess, theta_params)
        # wguess = 0.968167841907 #1.16
        # rguess = 0.06998690192 #.068
        # T_Hguess = 0.0304546765599 #0.046
        # factorguess = 274072.825051 #239344.894517
        ss_params_baseline = [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params]
        guesses = np.append([wguess, rguess, T_Hguess],np.append([BQ_guess.flatten()],np.append([theta_guess.flatten()],[factorguess])))
        [solutions_fsolve, infodict, ier, message] = opt.fsolve(SS_fsolve, guesses, args=ss_params_baseline, xtol=mindist_SS, full_output=True)
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        [wss, rss, T_Hss, factor_ss] = solutions_fsolve
        # wss = wguess
        # rss = rguess
        # T_Hss = T_Hguess
        # factor_ss = factorguess
        # fsolve_flag = False
        fsolve_flag = True
        # Return SS values of variables
        solution_params= [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params]
        output = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), wss, rss, T_Hss, factor_ss, solution_params, baseline, fsolve_flag)
    else:
        # baseline_ss_dir = os.path.join(
        #     baseline_dir, "SS/SS_vars.pkl")
        # ss_solutions = pickle.load(open(baseline_ss_dir, "rb"))
        # [wguess, rguess, T_Hguess, factor] = [ss_solutions['wss'], ss_solutions['rss'], ss_solutions['T_Hss'], ss_solutions['factor_ss']]
        wguess = START_VALUES['wss'] #0.968167841907 #1.16
        rguess = START_VALUES['rss'] #0.116998690192 #.068
        T_Hguess = START_VALUES['T_Hss'] #0.0304546765599 #0.046
        factor = START_VALUES['factor_ss'] #274072.825051 #239344.894517
        # wguess = 0.968167841907 #1.16
        # rguess = 0.086998690192 #.068
        # T_Hguess = 0.0304546765599 #0.046
        # factor = 225348.036701 #239344.894517
        ss_params_reform = [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params, factor]
        guesses = [wguess, rguess, T_Hguess]
        [solutions_fsolve, infodict, ier, message] = opt.fsolve(SS_fsolve_reform, guesses, args=ss_params_reform, xtol=mindist_SS, full_output=True)
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        # Return SS values of variables
        [wss, rss, T_Hss] = solutions_fsolve
        fsolve_flag = True
        # Return SS values of variables
        solution_params= [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params]
        output = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), wss, rss, T_Hss, factor, solution_params, baseline, fsolve_flag)

    return output
