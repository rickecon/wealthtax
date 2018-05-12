'''
------------------------------------------------------------------------
Last updated 4/9/2016

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            tax.py
            utils.py
            household.py
            firm.py
            OUTPUT/SS/ss_vars.pkl
            OUTPUT/Saved_moments/params_given.pkl
            OUTPUT/Saved_moments/params_changed.pkl


This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/TPIinit/TPIinit_vars.pkl
            OUTPUT/TPI/TPI_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.optimize as opt

import tax
import utils
import household
import firm
import os

TPI_START_VALUES = pickle.load(open("./OUTPUT_INCOME_REFORM/sigma2.0/TPI/TPI_vars.pkl", "rb"))

'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = True


'''
------------------------------------------------------------------------
Import steady state distribution, parameters and other objects from
steady state computation in ss_vars.pkl
------------------------------------------------------------------------
'''

def create_tpi_params(**sim_params):

    '''
    ------------------------------------------------------------------------
    Set factor and initial capital stock to SS from baseline
    ------------------------------------------------------------------------
    '''
    baseline_ss = os.path.join(sim_params['baseline_dir'], "SS/SS_vars.pkl")
    ss_baseline_vars = pickle.load(open(baseline_ss, "rb"))
    factor = ss_baseline_vars['factor_ss']
    initial_b = ss_baseline_vars['bssmat_splus1']
    if sim_params['baseline']:
        T_H_baseline = np.zeros(sim_params['T'] + sim_params['S'])
    else:
        baseline_tpi = os.path.join(sim_params['baseline_dir'], "TPI/TPI_vars.pkl")
        tpi_baseline_vars = pickle.load(open(baseline_tpi, "rb"))
        T_H_baseline = tpi_baseline_vars['T_H']

    if sim_params['baseline']==True:
        SS_values = (ss_baseline_vars['Kss'],ss_baseline_vars['Lss'], ss_baseline_vars['rss'],
                 ss_baseline_vars['wss'], ss_baseline_vars['BQss'], ss_baseline_vars['T_Hss'], ss_baseline_vars['Gss'],
                 ss_baseline_vars['bssmat_splus1'], ss_baseline_vars['nssmat'])
        wss = ss_baseline_vars['wss']
        nssmat = ss_baseline_vars['nssmat']
    elif sim_params['baseline']==False:
        reform_ss = os.path.join(sim_params['input_dir'], "SS/SS_vars.pkl")
        print('Directory for SS values for TPI = ', reform_ss)
        ss_reform_vars = pickle.load(open(reform_ss, "rb"))
        SS_values = (ss_reform_vars['Kss'],ss_reform_vars['Lss'], ss_reform_vars['rss'],
                 ss_reform_vars['wss'], ss_reform_vars['BQss'], ss_reform_vars['T_Hss'], ss_reform_vars['Gss'],
                 ss_reform_vars['bssmat_splus1'], ss_reform_vars['nssmat'])
        wss = ss_reform_vars['wss']
        nssmat = ss_reform_vars['nssmat']

    initial_n = nssmat  # set initial_n to SS value under policy regime running


    # Make a vector of all one dimensional parameters, to be used in the
    # following functions
    wealth_tax_params = [sim_params['h_wealth'], sim_params['p_wealth'], sim_params['m_wealth']]
    ellipse_params = [sim_params['b_ellipse'], sim_params['upsilon']]
    chi_params = [sim_params['chi_b_guess'], sim_params['chi_n_guess']]

    N_tilde = sim_params['omega'].sum(1) #this should just be one in each year given how we've constructed omega
    sim_params['omega'] = sim_params['omega'] / N_tilde.reshape(sim_params['T'] + sim_params['S'], 1)

    theta_params = (sim_params['e'], sim_params['S'], sim_params['retire'])
    theta = tax.replacement_rate_vals(nssmat, wss, factor, theta_params)

    tpi_params = [sim_params['J'], sim_params['S'], sim_params['T'], sim_params['BW'],
                  sim_params['beta'], sim_params['sigma'], sim_params['alpha'],
                  sim_params['Z'], sim_params['delta'], sim_params['ltilde'],
                  sim_params['nu'], sim_params['g_y'], sim_params['g_n_vector'],
                  sim_params['tau_payroll'], sim_params['tau_bq'], sim_params['rho'], sim_params['omega'], N_tilde,
                  sim_params['lambdas'], sim_params['imm_rates'], sim_params['e'],
                  sim_params['retire'], sim_params['mean_income_data'], factor, T_H_baseline] + \
                  wealth_tax_params + ellipse_params + chi_params + [theta]
    iterative_params = [sim_params['maxiter'], sim_params['mindist_SS'], sim_params['mindist_TPI']]

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, T_H_baseline, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params

    ## Assumption for tax functions is that policy in last year of BW is
    # extended permanently
    etr_params_TP = np.zeros((S,T+S,sim_params['etr_params'].shape[2]))
    etr_params_TP[:,:BW,:] = sim_params['etr_params']
    etr_params_TP[:,BW:,:] = np.reshape(sim_params['etr_params'][:,BW-1,:],(S,1,sim_params['etr_params'].shape[2]))

    mtrx_params_TP = np.zeros((S,T+S,sim_params['mtrx_params'].shape[2]))
    mtrx_params_TP[:,:BW,:] = sim_params['mtrx_params']
    mtrx_params_TP[:,BW:,:] = np.reshape(sim_params['mtrx_params'][:,BW-1,:],(S,1,sim_params['mtrx_params'].shape[2]))

    mtry_params_TP = np.zeros((S,T+S,sim_params['mtry_params'].shape[2]))
    mtry_params_TP[:,:BW,:] = sim_params['mtry_params']
    mtry_params_TP[:,BW:,:] = np.reshape(sim_params['mtry_params'][:,BW-1,:],(S,1,sim_params['mtry_params'].shape[2]))

    income_tax_params = (sim_params['analytical_mtrs'], etr_params_TP, mtrx_params_TP, mtry_params_TP)

    '''
    ------------------------------------------------------------------------
    Set other parameters and initial values
    ------------------------------------------------------------------------
    '''
    # Get an initial distribution of capital with the initial population
    # distribution

    b_sinit = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_b[:-1]))
    b_splus1init = initial_b


    omega_S_preTP = sim_params['omega_S_preTP']
    K0_params = (omega_S_preTP.reshape(S, 1), lambdas, imm_rates[0].reshape(S,1), g_n_vector[0], 'SS')
    K0 = household.get_K(initial_b, K0_params)


    initial_values = (K0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP)

    return (income_tax_params, tpi_params, iterative_params, initial_values, SS_values)


def firstdoughnutring(guesses, r, w, b, BQ, T_H, j, params):
    '''
    Solves the first entries of the upper triangle of the twist doughnut.  This is
    separate from the main TPI function because the the values of b and n are scalars,
    so it is easier to just have a separate function for these cases.
    Inputs:
        guesses = guess for b and n (2x1 list)
        winit = initial wage rate (scalar)
        rinit = initial rental rate (scalar)
        BQinit = initial aggregate bequest (scalar)
        T_H_init = initial lump sum tax (scalar)
        initial_b = initial distribution of capital (SxJ array)
        factor = steady state scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        parameters = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rates (Jx1 array)
    Output:
        euler errors (2x1 list)
    '''

    # unpack tuples of parameters
    income_tax_params, tpi_params, initial_b = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, T_H_baseline, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params


    b_splus1 = float(guesses[0])
    n = float(guesses[1])
    b_s = float(initial_b[-2, j])

    # Find errors from FOC for savings and FOC for labor supply
    retire_fd = 0  # this sets retire to true in these agents who are
    # in last period in life
    # Note using method = "SS" below because just for one period
    foc_save_params = (np.array([e[-1, j]]), sigma, beta, g_y, chi_b[j],
                       theta[j], tau_bq[j], rho[-1], lambdas[j], j, J,
                       S, analytical_mtrs,
                       np.reshape(etr_params[-1, 0, :],
                                  (1, etr_params.shape[2])),
                       np.reshape(mtry_params[-1, 0, :],
                                  (1, mtry_params.shape[2])), h_wealth,
                       p_wealth, m_wealth, tau_payroll, retire_fd, 'SS')
    error1 = household.FOC_savings(np.array([r]), np.array([w]), b_s,
                                   b_splus1, 0., np.array([n]),
                                   np.array([BQ]), factor,
                                   np.array([T_H]), foc_save_params)

    foc_labor_params = (np.array([e[-1, j]]), sigma, g_y, theta[j],
                        b_ellipse, upsilon, chi_n[-1], ltilde,
                        tau_bq[j], lambdas[j], j, J, S, analytical_mtrs,
                        np.reshape(etr_params[-1, 0, :],
                                   (1, etr_params.shape[2])),
                        np.reshape(mtrx_params[-1, 0, :],
                                   (1, mtrx_params.shape[2])), h_wealth,
                        p_wealth, m_wealth, tau_payroll, retire_fd,
                        'SS')
    error2 = household.FOC_labor(np.array([r]), np.array([w]), b_s,
                                 b_splus1, np.array([n]),
                                 np.array([BQ]), factor,
                                 np.array([T_H]), foc_labor_params)

    tax_params = (np.array([e[-1, j]]), lambdas[j], 'SS', retire_fd,
                  np.reshape(etr_params[-1, 0, :],
                             (1, etr_params.shape[2])), h_wealth,
                  p_wealth, m_wealth, tau_payroll, theta[j], tau_bq[j],
                  J, S)
    tax1 = tax.total_taxes(np.array([r]), np.array([w]), b_s,
                           np.array([n]), np.array([BQ]), factor,
                           np.array([T_H]), j, False, tax_params)
    cons_params = (np.array([e[-1, j]]), lambdas[j], g_y)
    cons = household.get_cons(np.array([r]), np.array([w]), b_s, b_splus1,
                    np.array([n]), np.array([BQ]), tax1, cons_params)

    if n < 0 or n > ltilde:
        error2 = 1e12
    if b_splus1 <= 0:
        error1 += 1e12
    # if cons <= 0:
    #     error1 += 1e12
    return [np.squeeze(error1)] + [np.squeeze(error2)]


def twist_doughnut(guesses, r, w, BQ, T_H, j, s, t, params):
    '''
    Parameters:
        guesses = distribution of capital and labor (various length list)
        w   = wage rate ((T+S)x1 array)
        r   = rental rate ((T+S)x1 array)
        BQ = aggregate bequests ((T+S)x1 array)
        T_H = lump sum tax over time ((T+S)x1 array)
        factor = scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        s = which upper triangle loop is being solved for (scalar)
        t = which diagonal is being solved for (scalar)
        params = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortalit rate (Sx1 array)
        lambdas = ability weights (Jx1 array)
        e = ability type (SxJ array)
        initial_b = capital stock distribution in period 0 (SxJ array)
        chi_b = chi^b_j (Jx1 array)
        chi_n = chi^n_s (Sx1 array)
    Output:
        Value of Euler error (various length list)
    '''

    income_tax_params, tpi_params, initial_b = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, T_H_baseline, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params

    length = len(guesses) / 2
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        b_s = np.array([(initial_b[-(s + 3), j])] + list(b_guess[:-1]))

    b_splus1 = b_guess
    b_splus2 = np.array(list(b_guess[1:]) + [0])
    w_s = w[t:t + length]
    w_splus1 = w[t + 1:t + length + 1]
    r_s = r[t:t + length]
    r_splus1 = r[t + 1:t + length + 1]
    n_s = n_guess
    n_extended = np.array(list(n_guess[1:]) + [0])
    e_s = e[-length:, j]
    e_extended = np.array(list(e[-length + 1:, j]) + [0])
    BQ_s = BQ[t:t + length]
    BQ_splus1 = BQ[t + 1:t + length + 1]
    T_H_s = T_H[t:t + length]
    T_H_splus1 = T_H[t + 1:t + length + 1]

# Errors from FOC for savings
    foc_save_params = (e_s, sigma, beta, g_y, chi_b[j], theta, tau_bq,
                       rho[-(length):], lambdas[j], j, J, S,
                       analytical_mtrs, etr_params, mtry_params,
                       h_wealth, p_wealth, m_wealth, tau_payroll,
                       retire, 'TPI')
    error1 = household.FOC_savings(r_s, w_s, b_s, b_splus1, b_splus2,
                                   n_s, BQ_s, factor, T_H_s,
                                   foc_save_params)

    # Errors from FOC for labor supply
    foc_labor_params = (e_s, sigma, g_y, theta, b_ellipse, upsilon,
                        chi_n[-length:], ltilde, tau_bq, lambdas[j], j,
                        J, S, analytical_mtrs, etr_params, mtrx_params,
                        h_wealth, p_wealth, m_wealth, tau_payroll,
                        retire, 'TPI')
    error2 = household.FOC_labor(r_s, w_s, b_s, b_splus1, n_s, BQ_s,
                                 factor, T_H_s, foc_labor_params)

    tax_params = (e_s, lambdas[j], 'TPI', retire, etr_params, h_wealth,
                  p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax_s = tax.total_taxes(r_s, w_s, b_s, n_s, BQ_s, factor, T_H_s, j,
                            False, tax_params)
    cons_params = (e_s, lambdas[j], g_y)
    cons_s = household.get_cons(r_s, w_s, b_s, b_splus1, n_s, BQ_s, tax_s,
                      cons_params)

    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] = 1e12
    mask2 = n_guess > ltilde
    error2[mask2] = 1e12
    # mask3 = cons_s < 0
    # error2[mask3] += 1e12
    mask4 = b_guess <= 0
    error2[mask4] += 1e12
    # mask5 = cons_splus1 < 0
    mask5 = b_splus1 < 0
    error2[mask5] += 1e12
    return list(error1.flatten()) + list(error2.flatten())


def inner_loop(guesses, outer_loop_vars, params):
    '''
    Solves inner loop of TPI.  Given path of economic aggregates and factor prices, solves
    househld problem

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        T_H        = [T,] vector, lump sum transfer amount(s)


    Functions called:
        firstdoughnutring()
        twist_doughnut()

    Objects in function:


    Returns: euler_errors, b_mat, n_mat

    '''
    #unpack variables and parameters pass to function
    income_tax_params, tpi_params, initial_values, ind = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, T_H_baseline, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params
    K0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP = initial_values

    guesses_b, guesses_n = guesses
    r, w, BQ, T_H = outer_loop_vars

    # initialize arrays
    b_mat = np.zeros((T + S, S, J))
    n_mat = np.zeros((T + S, S, J))
    euler_errors = np.zeros((T, 2 * S, J))
    euler_errors_n = np.zeros((T + S, S, J))
    euler_errors_b = np.zeros((T + S, S, J))

    for j in xrange(J):
        first_doughnut_params = (income_tax_params, tpi_params, initial_b)
        first_guesses = [guesses_b[0, -1, j], guesses_n[0, -1, j]]
        # if j == 0:
        #     first_guesses = [guesses_b[0, -1, j], guesses_n[0, -1, j]]
        # else:
        #     first_guesses = [guesses_b[0, -1, j - 1], guesses_n[0, -1, j - 1]]
        root_result =\
            opt.root(firstdoughnutring, first_guesses,
                     args=(r[0], w[0], initial_b, BQ[0, j], T_H[0], j,
                           first_doughnut_params), method='lm',
                     tol=MINIMIZER_TOL)
        b_mat[0, -1, j], n_mat[0, -1, j] = root_result.x
        euler_errors_b[0, -1, j], euler_errors_n[0, -1, j] = root_result.fun



        # print 'J= ', j
        # print 'first donut errors: ', np.absolute(infodict['fvec']).max()

        for s in xrange(S - 2):  # Upper triangle
            ind2 = np.arange(s + 2)

            # initialize array of diagonal elements
            length_diag = (np.diag(np.transpose(etr_params[:, :S, 0]), S-(s+2))).shape[0]
            etr_params_to_use = np.zeros((length_diag, etr_params.shape[2]))
            mtrx_params_to_use = np.zeros((length_diag, mtrx_params.shape[2]))
            mtry_params_to_use = np.zeros((length_diag, mtry_params.shape[2]))
            for i in range(etr_params.shape[2]):
                etr_params_to_use[:, i] = np.diag(np.transpose(etr_params[:, :S, i]), S-(s+2))
                mtrx_params_to_use[:, i] = np.diag(np.transpose(mtrx_params[:, :S, i]), S-(s+2))
                mtry_params_to_use[:, i] = np.diag(np.transpose(mtry_params[:, :S, i]), S-(s+2))


            inc_tax_params_upper = (analytical_mtrs, etr_params_to_use,
                                    mtrx_params_to_use, mtry_params_to_use)

            TPI_solver_params = (inc_tax_params_upper, tpi_params, initial_b)

            b_guesses_to_use = np.diag(guesses_b[:S, :, j], S - (s + 2))
            n_guesses_to_use = np.diag(guesses_n[:S, :, j], S - (s + 2))
            # if j == 0:
            #     b_guesses_to_use = np.diag(
            #         guesses_b[:S, :, j], S - (s + 2))
            #     n_guesses_to_use = np.diag(guesses_n[:S, :, j], S - (s + 2))
            # else:
            #     b_guesses_to_use = np.diag(
            #         guesses_b[:S, :, j - 1], S - (s + 2))
            #     n_guesses_to_use = np.diag(guesses_n[:S, :, j - 1], S - (s + 2))
            twist_guesses = list(b_guesses_to_use) + list(n_guesses_to_use)
            root_result =\
                opt.root(twist_doughnut, twist_guesses,
                         args=(r, w, BQ[:, j], T_H, j, s, 0, TPI_solver_params),
                         method='lm', tol=MINIMIZER_TOL)
            solutions = root_result.x
            euler_errors_b[ind2,  S - (s + 2) + ind2, j] = root_result.fun[:len(solutions) / 2]
            euler_errors_n[ind2,  S - (s + 2) + ind2, j] = root_result.fun[len(solutions) / 2:]


            # print 'J= ', j
            # print 'S = ', s
            # print 'twist donut errors: ', np.absolute(infodict['fvec']).max()

            b_vec = solutions[:len(solutions) / 2]
            b_mat[ind2, S - (s + 2) + ind2, j] = b_vec
            n_vec = solutions[len(solutions) / 2:]
            n_mat[ind2, S - (s + 2) + ind2, j] = n_vec


        for t in xrange(0, T):
            # b_guesses_to_use = .75 * \
            #     np.diag(guesses_b[t + 1:t + S + 1, :, j])
            # b_guesses_to_use = .75 * \
            #     np.diag(guesses_b[t:t + S, :, j])
            # n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j])

            # initialize array of diagonal elements
            length_diag = (np.diag(np.transpose(etr_params[:, t:t+S, 0]))).shape[0]
            etr_params_to_use = np.zeros((length_diag, etr_params.shape[2]))
            mtrx_params_to_use = np.zeros((length_diag, mtrx_params.shape[2]))
            mtry_params_to_use = np.zeros((length_diag, mtry_params.shape[2]))
            for i in range(etr_params.shape[2]):
                etr_params_to_use[:, i] = np.diag(np.transpose(etr_params[:, t:t+S, i]))
                mtrx_params_to_use[:, i] = np.diag(np.transpose(mtrx_params[:, t:t+S, i]))
                mtry_params_to_use[:, i] = np.diag(np.transpose(mtry_params[:, t:t+S, i]))

            inc_tax_params_TP = (analytical_mtrs, etr_params_to_use,
                                 mtrx_params_to_use, mtry_params_to_use)


            TPI_solver_params = (inc_tax_params_TP, tpi_params, None)
            # [solutions, infodict, ier, message] = opt.fsolve(twist_doughnut, list(
            #     b_guesses_to_use) + list(n_guesses_to_use), args=(
            #     r, w, BQ[:, j], T_H, j, None, t, TPI_solver_params), xtol=MINIMIZER_TOL, full_output=True)
            # euler_errors[t, :, j] = infodict['fvec']

            b_guesses_to_use = 1.0 * np.diag(guesses_b[t:t + S, :, j])
            n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j])
            # if j == 0:
            #     b_guesses_to_use = 1.0 * np.diag(guesses_b[t:t + S, :, j])
            #     n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j])
            # else:
            #     b_guesses_to_use = 1.0 * np.diag(guesses_b[t:t + S, :, j - 1])
            #     n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j - 1])
            twist_guesses = list(b_guesses_to_use) + list(n_guesses_to_use)
            root_result =\
                opt.root(twist_doughnut, twist_guesses,
                           args=(r, w, BQ[:, j], T_H, j, None, t,
                                 TPI_solver_params), method='lm',
                           tol=MINIMIZER_TOL)
            solutions = root_result.x
            euler_errors[t, :, j] = root_result.fun
            euler_errors_b[t, :, j] = root_result.fun[:S]
            euler_errors_n[t, :, j] = root_result.fun[S:]

            # print 'J= ', j
            # print 't = ', t
            # print 'twist donut #2 errors: ', np.absolute(infodict['fvec']).max()

            b_vec = solutions[:S]
            b_mat[t + ind, ind, j] = b_vec
            n_vec = solutions[S:]
            n_mat[t + ind, ind, j] = n_vec

        print 'j = ', j
        # print 'inner loop euler errors: ', np.absolute(euler_errors).max()
        print 'max savings euler errors: ', np.absolute(euler_errors_b).max()
        print 'max labor euler errors: ', np.absolute(euler_errors_n).max()

    return euler_errors, b_mat, n_mat


def run_TPI(income_tax_params, tpi_params, iterative_params,
            initial_values, SS_values, fix_transfers=False,
            output_dir="./OUTPUT"):

    # unpack tuples of parameters
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    maxiter, mindist_SS, mindist_TPI = iterative_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, T_H_baseline, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params
    K0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP = initial_values
    Kss, Lss, rss, wss, BQss, T_Hss, Gss, bssmat_splus1, nssmat = SS_values


    TPI_FIG_DIR = output_dir
    # Initialize guesses at time paths
    domain = np.linspace(0, T, T)
    # r = np.ones(T + S) * rss
    # BQ = np.zeros((T + S, J))
    # BQ0_params = (omega_S_preTP.reshape(S, 1), lambdas, rho.reshape(S, 1), g_n_vector[0], 'SS')
    # BQ0 = household.get_BQ(r[0], initial_b, BQ0_params)
    # for j in xrange(J):
    #     BQ[:, j] = list(np.linspace(BQ0[j], BQss[j], T)) + [BQss[j]] * S
    # BQ = np.array(BQ)
    # # print "BQ values = ", BQ[0, :], BQ[100, :], BQ[-1, :], BQss
    # # print "K0 vs Kss = ", K0-Kss
    #
    # if fix_transfers:
    #     T_H = T_H_baseline
    # else:
    #     if np.abs(T_Hss) < 1e-13 :
    #         T_Hss2 = 0.0 # sometimes SS is very small but not zero, even if taxes are zero, this get's rid of the approximation error, which affects the perc changes below
    #     else:
    #         T_Hss2 = T_Hss
    #     T_H = np.ones(T + S) * T_Hss2 * (r/rss)
    # G = np.ones(T + S) * Gss
    # # print "T_H values = ", T_H[0], T_H[100], T_H[-1], T_Hss
    # # print "omega diffs = ", (omega_S_preTP-omega[-1]).max(), (omega[10]-omega[-1]).max()
    #
    # Make array of initial guesses for labor supply and savings
    domain2 = np.tile(domain.reshape(T, 1, 1), (1, S, J))
    ending_b = bssmat_splus1
    guesses_b = (-1 / (domain2 + 1)) * (ending_b - initial_b) + ending_b
    ending_b_tail = np.tile(ending_b.reshape(1, S, J), (S, 1, 1))
    guesses_b = np.append(guesses_b, ending_b_tail, axis=0)
    # print 'diff btwn start and end b: ', (guesses_b[0]-guesses_b[-1]).max()
    #
    domain3 = np.tile(np.linspace(0, 1, T).reshape(T, 1, 1), (1, S, J))
    guesses_n = domain3 * (nssmat - initial_n) + initial_n
    ending_n_tail = np.tile(nssmat.reshape(1, S, J), (S, 1, 1))
    guesses_n = np.append(guesses_n, ending_n_tail, axis=0)
    # b_mat = np.zeros((T + S, S, J))
    # n_mat = np.zeros((T + S, S, J))
    ind = np.arange(S)
    # # print 'diff btwn start and end n: ', (guesses_n[0]-guesses_n[-1]).max()
    #
    # # find economic aggregates
    # K = np.zeros(T+S)
    # L = np.zeros(T+S)
    # K[0] = K0
    # K_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J), imm_rates[:T-1].reshape(T-1,S,1), g_n_vector[1:T], 'TPI')
    # K[1:T] = household.get_K(guesses_b[:T-1], K_params)
    # K[T:] = Kss
    # L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI')
    # L[:T] = firm.get_L(guesses_n[:T], L_params)
    # L[T:] = Lss
    # Y_params = (alpha, Z)
    # Y = firm.get_Y(K, L, Y_params)
    # r_params = (alpha, delta)
    # r[:T] = firm.get_r(Y[:T], K[:T], r_params)

    r = TPI_START_VALUES['r']
    K = TPI_START_VALUES['K']
    L = TPI_START_VALUES['L']
    Y = TPI_START_VALUES['Y']
    T_H = TPI_START_VALUES['T_H']
    BQ = TPI_START_VALUES['BQ']
    G = TPI_START_VALUES['G']
    # guesses_b = TPI_START_VALUES['b_mat']
    # guesses_n = TPI_START_VALUES['n_mat']


    TPIiter = 0
    TPIdist = 10
    PLOT_TPI = False

    euler_errors = np.zeros((T, 2 * S, J))
    TPIdist_vec = np.zeros(maxiter)

    # print 'analytical mtrs in tpi = ', analytical_mtrs

    while (TPIiter < maxiter) and (TPIdist >= mindist_TPI):
        # Plot TPI for K for each iteration, so we can see if there is a
        # problem
        if PLOT_TPI is True:
            K_plot = list(K) + list(np.ones(10) * Kss)
            L_plot = list(L) + list(np.ones(10) * Lss)
            plt.figure()
            plt.axhline(
                y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
            plt.plot(np.arange(
                T + 10), Kpath_plot[:T + 10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
            plt.savefig(os.path.join(TPI_FIG_DIR, "TPI_K"))


        guesses = (guesses_b, guesses_n)
        w_params = (Z, alpha, delta)
        w = firm.get_w_from_r(r, w_params)
        # print 'r and rss diff = ', r-rss
        # print 'w and wss diff = ', w-wss
        # print 'BQ and BQss diff = ', BQ-BQss
        # print 'T_H and T_Hss diff = ', T_H - T_Hss
        # print 'guess b and bss = ', (bssmat_splus1 - guesses_b).max()
        # print 'guess n and nss = ', (nssmat - guesses_n).max()
        outer_loop_vars = (r, w, BQ, T_H)
        inner_loop_params = (income_tax_params, tpi_params, initial_values, ind)

        # Solve HH problem in inner loop
        euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)

        # print 'guess b and bss = ', (b_mat - guesses_b).max()
        # print 'guess n and nss over time = ', (n_mat - guesses_n).max(axis=2).max(axis=1)
        # print 'guess n and nss over age = ', (n_mat - guesses_n).max(axis=0).max(axis=1)
        # print 'guess n and nss over ability = ', (n_mat - guesses_n).max(axis=0).max(axis=0)
        # quit()

        print 'Max Euler error: ', (np.abs(euler_errors)).max()

        bmat_s = np.zeros((T, S, J))
        bmat_s[0, 1:, :] = initial_b[:-1, :]
        bmat_s[1:, 1:, :] = b_mat[:T-1, :-1, :]
        bmat_splus1 = np.zeros((T, S, J))
        bmat_splus1[:, :, :] = b_mat[:T, :, :]

        K[0] = K0
        K_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J),
                    imm_rates[:T-1].reshape(T-1, S, 1), g_n_vector[1:T], 'TPI')
        K[1:T] = household.get_K(bmat_splus1[:T-1], K_params)
        L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1),
                    lambdas.reshape(1, 1, J), 'TPI')
        L[:T] = firm.get_L(n_mat[:T], L_params)
        # print 'K diffs = ', K-K0
        # print 'L diffs = ', L-L[0]

        Y_params = (alpha, Z)
        Ynew = firm.get_Y(K[:T], L[:T], Y_params)
        r_params = (alpha, delta)
        rnew = firm.get_r(Ynew[:T], K[:T], r_params)
        wnew = firm.get_w_from_r(rnew, w_params)

        omega_shift = np.append(omega_S_preTP.reshape(1, S),
                                omega[:T-1, :], axis=0)
        BQ_params = (omega_shift.reshape(T, S, 1), lambdas.reshape(1, 1, J),
                     rho.reshape(1, S, 1), g_n_vector[:T].reshape(T, 1), 'TPI')
        # b_mat_shift = np.append(np.reshape(initial_b, (1, S, J)),
        #                         b_mat[:T-1, :, :], axis=0)
        b_mat_shift = bmat_splus1[:T, :, :]
        # print 'b diffs = ', (bmat_splus1[100, :, :] - initial_b).max(), (bmat_splus1[0, :, :] - initial_b).max(), (bmat_splus1[1, :, :] - initial_b).max()
        # print 'r diffs = ', rnew[1]-r[1], rnew[100]-r[100], rnew[-1]-r[-1]
        BQnew = household.get_BQ(rnew[:T].reshape(T, 1), b_mat_shift,
                                 BQ_params)
        BQss2 = np.empty(J)
        for j in range(J):
            BQss_params = (omega[1, :], lambdas[j], rho, g_n_vector[1], 'SS')
            BQss2[j] = household.get_BQ(rnew[1], bmat_splus1[1, :, j],
                                        BQss_params)
        # print 'BQ test = ', BQss2-BQss, BQss-BQnew[1], BQss-BQnew[100], BQss-BQnew[-1]

        total_tax_params = np.zeros((T, S, J, etr_params.shape[2]))
        for i in range(etr_params.shape[2]):
            total_tax_params[:, :, :, i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

        tax_receipt_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
                total_tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J)
        net_tax_receipts = np.array(list(tax.get_lump_sum(np.tile(rnew[:T].reshape(T, 1, 1),(1,S,J)), np.tile(wnew[:T].reshape(T, 1, 1),(1,S,J)),
               bmat_s, n_mat[:T,:,:], BQnew[:T].reshape(T, 1, J), factor, tax_receipt_params)) + [T_Hss] * S)

        r[:T] = utils.convex_combo(rnew[:T], r[:T], nu)
        BQ[:T] = utils.convex_combo(BQnew[:T], BQ[:T], nu)
        if fix_transfers:
            T_H_new = T_H
            G[:T] = net_tax_receipts[:T] - T_H[:T]
        else:
            T_H_new = net_tax_receipts
            T_H[:T] = utils.convex_combo(T_H_new[:T], T_H[:T], nu)
            G[:T] = 0.0

        etr_params_path = np.zeros((T,S,J,etr_params.shape[2]))
        for i in range(etr_params.shape[2]):
            etr_params_path[:,:,:,i] = np.tile(
                np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))
        tax_path_params = (np.tile(e.reshape(1, S, J),(T,1,1)),
                           lambdas, 'TPI', retire, etr_params_path, h_wealth,
                           p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
        b_to_use = np.zeros((T, S, J))
        b_to_use[0, 1:, :] = initial_b[:-1, :]
        b_to_use[1:, 1:, :] = b_mat[:T-1, :-1, :]
        tax_path = tax.total_taxes(
            np.tile(r[:T].reshape(T, 1, 1),(1,S,J)),
            np.tile(w[:T].reshape(T, 1, 1),(1,S,J)), b_to_use,
            n_mat[:T,:,:], BQ[:T, :].reshape(T, 1, J), factor,
            T_H[:T].reshape(T, 1, 1), None, False, tax_path_params)

        y_path = (np.tile(r[:T].reshape(T, 1, 1), (1, S, J)) * b_to_use[:T, :, :] +
                  np.tile(w[:T].reshape(T, 1, 1), (1, S, J)) *
                  np.tile(e.reshape(1, S, J), (T, 1, 1)) * n_mat[:T, :, :])
        cons_params = (e.reshape(1, S, J), lambdas.reshape(1, 1, J), g_y)
        c_path = household.get_cons(r[:T].reshape(T, 1, 1), w[:T].reshape(T, 1, 1), b_to_use[:T,:,:], b_mat[:T,:,:], n_mat[:T,:,:],
                       BQ[:T].reshape(T, 1, J), tax_path, cons_params)


        guesses_b = utils.convex_combo(b_mat, guesses_b, nu)
        guesses_n = utils.convex_combo(n_mat, guesses_n, nu)
        if T_H.all() != 0:
            TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) +
                               list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) +
                               list(utils.pct_diff_func(T_H_new[:T], T_H[:T]))).max()
            print 'r dist = ', np.array(list(utils.pct_diff_func(rnew[:T], r[:T]))).max()
            print 'BQ dist = ', np.array(list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten())).max()
            print 'T_H dist = ', np.array(list(utils.pct_diff_func(T_H_new[:T], T_H[:T]))).max()
            # print 'r old = ', r[:T]
            # print 'r new = ', rnew[:T]
            # print 'K old = ', K[:T]
            # print 'L old = ', L[:T]
            # print 'income = ', y_path[:, :, -1]
            # print 'taxes = ', tax_path[:, :, -1]
            # print 'labor supply = ', n_mat[:, :, -1]
            # print 'max and min labor = ', n_mat.max(), n_mat.min()
            # print 'max and min labor = ', np.argmax(n_mat), np.argmin(n_mat)
            # print 'max and min labor, j = 7 = ', n_mat[:,:,-1].max(), n_mat[:,:,-1].min()
            # print 'max and min labor, j = 6 = ', n_mat[:,:,-2].max(), n_mat[:,:,-2].min()
            # print 'max and min labor, j = 5 = ', n_mat[:,:,4].max(), n_mat[:,:,4].min()
            # print 'max and min labor, j = 4 = ', n_mat[:,:,3].max(), n_mat[:,:,3].min()
            # print 'max and min labor, j = 3 = ', n_mat[:,:,2].max(), n_mat[:,:,2].min()
            # print 'max and min labor, j = 2 = ', n_mat[:,:,1].max(), n_mat[:,:,1].min()
            # print 'max and min labor, j = 1 = ', n_mat[:,:,0].max(), n_mat[:,:,0].min()
            # print 'max and min labor, S = 80 = ', n_mat[:,-1,-1].max(), n_mat[:,-1,-1].min()
            # print "number  > 1 = ", (n_mat > 1).sum()
            # print "number  < 0, = ", (n_mat < 0).sum()
            # print "number  > 1, j=7 = ", (n_mat[:T,:,-1] > 1).sum()
            # print "number  < 0, j=7 = ", (n_mat[:T,:,-1] < 0).sum()
            # print "number  > 1, s=80, j=7 = ", (n_mat[:T,-1,-1] > 1).sum()
            # print "number  < 0, s=80, j=7 = ", (n_mat[:T,-1,-1] < 0).sum()
            # print "number  > 1, j= 7, age 80= ", (n_mat[:T,-1,-1] > 1).sum()
            # print "number  < 0, j = 7, age 80= ", (n_mat[:T,-1,-1] < 0).sum()
            # print "number  > 1, j= 7, age 80, period 0 to 10= ", (n_mat[:30,-1,-1] > 1).sum()
            # print "number  < 0, j = 7, age 80, period 0 to 10= ", (n_mat[:30,-1,-1] < 0).sum()
            # print "number  > 1, j= 7, age 70-79, period 0 to 10= ", (n_mat[:30,70:80,-1] > 1).sum()
            # print "number  < 0, j = 7, age 70-79, period 0 to 10= ", (n_mat[:30,70:80   ,-1] < 0).sum()
            # diag_dict = {'n_mat': n_mat, 'b_mat': b_mat, 'y_path': y_path, 'c_path': c_path}
            # pickle.dump(diag_dict, open('tpi_iter1.pkl', 'wb'))

        else:
            TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) +
                               list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) +
                               list(np.abs(T_H_new[:T]-T_H[:T]))).max()
        TPIdist_vec[TPIiter] = TPIdist
        # After T=10, if cycling occurs, drop the value of nu
        # wait til after T=10 or so, because sometimes there is a jump up
        # in the first couple iterations
        # if TPIiter > 10:
        #     if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter - 1] > 0:
        #         nu /= 2
        #         print 'New Value of nu:', nu
        TPIiter += 1
        print '\tIteration:', TPIiter
        print '\t\tDistance:', TPIdist

    Y[:T] = Ynew


    # Solve HH problem in inner loop
    guesses = (guesses_b, guesses_n)
    outer_loop_vars = (r, w, BQ, T_H)
    inner_loop_params = (income_tax_params, tpi_params, initial_values, ind)
    euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)

    bmat_s = np.zeros((T, S, J))
    bmat_s[0, 1:, :] = initial_b[:-1, :]
    bmat_s[1:, 1:, :] = b_mat[:T-1, :-1, :]
    bmat_splus1 = np.zeros((T, S, J))
    bmat_splus1[:, :, :] = b_mat[:T, :, :]

    K[0] = K0
    K_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J), imm_rates[:T-1].reshape(T-1,S,1), g_n_vector[1:T], 'TPI')
    K[1:T] = household.get_K(bmat_splus1[:T-1], K_params)
    L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI')
    L[:T]  = firm.get_L(n_mat[:T], L_params)

    Y_params = (alpha, Z)
    Ynew = firm.get_Y(K[:T], L[:T], Y_params)
    r_params = (alpha, delta)
    rnew = firm.get_r(Ynew[:T], K[:T], r_params)
    wnew = firm.get_w_from_r(rnew, w_params)

    omega_shift = np.append(omega_S_preTP.reshape(1,S),omega[:T-1,:],axis=0)
    BQ_params = (omega_shift.reshape(T, S, 1), lambdas.reshape(1, 1, J), rho.reshape(1, S, 1),
                 g_n_vector[:T].reshape(T, 1), 'TPI')
    b_mat_shift = np.append(np.reshape(initial_b,(1,S,J)),b_mat[:T-1,:,:],axis=0)
    BQnew = household.get_BQ(rnew[:T].reshape(T, 1), b_mat_shift, BQ_params)

    total_tax_params = np.zeros((T,S,J,etr_params.shape[2]))
    for i in range(etr_params.shape[2]):
        total_tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

    tax_receipt_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
            total_tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J)
    net_tax_receipts = np.array(list(tax.get_lump_sum(np.tile(rnew[:T].reshape(T, 1, 1),(1,S,J)), np.tile(wnew[:T].reshape(T, 1, 1),(1,S,J)),
           bmat_s, n_mat[:T,:,:], BQnew[:T].reshape(T, 1, J), factor, tax_receipt_params)) + [T_Hss] * S)

    if fix_transfers:
        G[:T] = net_tax_receipts[:T] - T_H[:T]
    else:
        T_H[:T] = net_tax_receipts[:T]
        G[:T] = 0.0

    etr_params_path = np.zeros((T,S,J,etr_params.shape[2]))
    for i in range(etr_params.shape[2]):
        etr_params_path[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))
    tax_path_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas, 'TPI', retire, etr_params_path, h_wealth,
                       p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax_path = tax.total_taxes(np.tile(r[:T].reshape(T, 1, 1),(1,S,J)), np.tile(w[:T].reshape(T, 1, 1),(1,S,J)), bmat_s,
                               n_mat[:T,:,:], BQ[:T, :].reshape(T, 1, J), factor, T_H[:T].reshape(T, 1, 1), None, False, tax_path_params)

    cons_params = (e.reshape(1, S, J), lambdas.reshape(1, 1, J), g_y)
    c_path = household.get_cons(r[:T].reshape(T, 1, 1), w[:T].reshape(T, 1, 1), bmat_s, bmat_splus1, n_mat[:T,:,:],
                   BQ[:T].reshape(T, 1, J), tax_path, cons_params)
    C_params = (omega[:T].reshape(T, S, 1), lambdas, 'TPI')
    C = household.get_C(c_path, C_params)
    I_params = (delta, g_y, omega[:T].reshape(T, S, 1), lambdas, imm_rates[:T].reshape(T, S, 1), g_n_vector[1:T+1], 'TPI')
    I = firm.get_I(bmat_splus1[:T], K[1:T+1], K[:T], I_params)
    rc_error = Y[:T] - C[:T] - I[:T] - G[:T]
    print 'Resource Constraint Difference:', rc_error

    # compute utility
    u_params = (sigma, np.tile(chi_n.reshape(1, S, 1), (T, 1, J)),
                b_ellipse, ltilde, upsilon,
                np.tile(rho.reshape(1, S, 1), (T, 1, J)),
                np.tile(chi_b.reshape(1, 1, J), (T, S, 1)))
    utility_path = household.get_u(c_path[:T, :, :], n_mat[:T, :, :],
                                   bmat_splus1[:T, :, :], u_params)

    # compute before and after-tax income
    y_path = (np.tile(r[:T].reshape(T, 1, 1), (1, S, J)) * bmat_s[:T, :, :] +
              np.tile(w[:T].reshape(T, 1, 1), (1, S, J)) *
              np.tile(e.reshape(1, S, J), (T, 1, 1)) * n_mat[:T, :, :])
    inctax_params = (np.tile(e.reshape(1, S, J), (T, 1, 1)), etr_params_path)
    y_aftertax_path = (y_path -
                       tax.tau_income(np.tile(r[:T].reshape(T, 1, 1), (1, S, J)),
                                      np.tile(w[:T].reshape(T, 1, 1), (1, S, J)),
                                      bmat_s[:T,:,:], n_mat[:T,:,:], factor, inctax_params))

    # compute after-tax wealth
    wtax_params = (h_wealth, p_wealth, m_wealth)
    b_aftertax_path = bmat_s[:T,:,:] - tax.tau_wealth(bmat_s[:T,:,:], wtax_params)

    print'Checking time path for violations of constaints.'
    for t in xrange(T):
        household.constraint_checker_TPI(
            b_mat[t], n_mat[t], c_path[t], t, ltilde)

    eul_savings = euler_errors[:, :S, :].max(1).max(1)
    eul_laborleisure = euler_errors[:, S:, :].max(1).max(1)

    print 'Max Euler error, savings: ', eul_savings
    print 'Max Euler error labor supply: ', eul_laborleisure



    '''
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    '''

    output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I, 'BQ': BQ, 'G': G,
              'T_H': T_H, 'r': r, 'w': w, 'b_mat': b_mat, 'n_mat': n_mat,
              'c_path': c_path, 'tax_path': tax_path, 'bmat_s': bmat_s,
              'utility_path': utility_path, 'b_aftertax_path': b_aftertax_path,
              'y_aftertax_path': y_aftertax_path, 'y_path': y_path,
              'eul_savings': eul_savings, 'eul_laborleisure': eul_laborleisure}

    macro_output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I,
                    'BQ': BQ, 'G': G, 'T_H': T_H, 'r': r, 'w': w,
                    'tax_path': tax_path}


    # if ((TPIiter >= maxiter) or (np.absolute(TPIdist) > mindist_TPI)) and ENFORCE_SOLUTION_CHECKS :
    #     raise RuntimeError("Transition path equlibrium not found")
    #
    # if ((np.any(np.absolute(rc_error) >= 1e-6))
    #     and ENFORCE_SOLUTION_CHECKS):
    #     raise RuntimeError("Transition path equlibrium not found")
    #
    # if ((np.any(np.absolute(eul_savings) >= mindist_TPI) or
    #     (np.any(np.absolute(eul_laborleisure) > mindist_TPI)))
    #     and ENFORCE_SOLUTION_CHECKS):
    #     raise RuntimeError("Transition path equlibrium not found")

    return output, macro_output
