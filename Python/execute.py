'''
A 'smoke test' for the ogusa package. Uses a fake data set to run the
baseline
'''

import cPickle as pickle
import os
import numpy as np
import time

import ogusa
from ogusa import calibrate
ogusa.parameters.DATASET = 'REAL'


def runner(output_base, baseline_dir, baseline=False, analytical_mtrs=True,
           age_specific=False, reform=0, user_params={}, guid='',
           run_micro=True, calibrate_model=False):

    #from ogusa import parameters, wealth, labor, demographics, income
    from ogusa import parameters, demographics, income, utils

    tick = time.time()

    #Create output directory structure
    saved_moments_dir = os.path.join(output_base, "Saved_moments")
    ss_dir = os.path.join(output_base, "SS")
    tpi_dir = os.path.join(output_base, "TPI")
    dirs = [saved_moments_dir, ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print "making dir: ", _dir
            os.makedirs(_dir)
        except OSError as oe:
            pass

    print ("in runner, baseline is ", baseline)
    run_params = ogusa.parameters.get_parameters(baseline=baseline, reform=reform,
                          guid=guid, user_modifiable=True)
    run_params['analytical_mtrs'] = analytical_mtrs

    # Modify ogusa parameters based on user input
    if 'frisch' in user_params:
        print "updating fricsh and associated"
        b_ellipse, upsilon = ogusa.elliptical_u_est.estimation(user_params['frisch'],
                                                               run_params['ltilde'])
        run_params['b_ellipse'] = b_ellipse
        run_params['upsilon'] = upsilon
        run_params.update(user_params)

    # Modify ogusa parameters based on user input
    if 'sigma' in user_params:
        print "updating sigma"
        run_params['sigma'] = user_params['sigma']
        run_params.update(user_params)


    from ogusa import SS, TPI


    calibrate_model = False
    # List of parameter names that will not be changing (unless we decide to
    # change them for a tax experiment)

    param_names = ['S', 'J', 'T', 'BW', 'lambdas', 'starting_age', 'ending_age',
                'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
                'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
                'analytical_mtrs', 'b_ellipse', 'k_ellipse', 'upsilon',
                'chi_b_guess', 'chi_n_guess','etr_params','mtrx_params',
                'mtry_params','tau_payroll', 'tau_bq',
                'retire', 'mean_income_data', 'g_n_vector',
                'h_wealth', 'p_wealth', 'm_wealth',
                'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'imm_rates','e', 'rho', 'omega_S_preTP']

    '''
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    '''

    sim_params = {}
    for key in param_names:
        sim_params[key] = run_params[key]

    sim_params['output_dir'] = output_base
    sim_params['run_params'] = run_params

    income_tax_params, ss_parameters, iterative_params, chi_params = SS.create_steady_state_parameters(**sim_params)

    ss_outputs = SS.run_SS(income_tax_params, ss_parameters, iterative_params, chi_params, baseline,
                                     baseline_dir=baseline_dir)

    '''
    ------------------------------------------------------------------------
        Pickle SS results
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))


    '''
    ------------------------------------------------------------------------
        Run the TPI simulation
    ------------------------------------------------------------------------
    '''

    sim_params['baseline'] = baseline
    sim_params['input_dir'] = output_base
    sim_params['baseline_dir'] = baseline_dir


    income_tax_params, tpi_params, iterative_params, initial_values, SS_values = TPI.create_tpi_params(**sim_params)

    tpi_output, macro_output = TPI.run_TPI(income_tax_params,
        tpi_params, iterative_params, initial_values, SS_values, output_dir=output_base)


    '''
    ------------------------------------------------------------------------
        Pickle TPI results
    ------------------------------------------------------------------------
    '''
    tpi_dir = os.path.join(output_base, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
    pickle.dump(tpi_output, open(tpi_vars, "wb"))

    tpi_dir = os.path.join(output_base, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_macro_vars.pkl")
    pickle.dump(macro_output, open(tpi_vars, "wb"))


    print "Time path iteration complete.  It"
    print "took {0} seconds to get that part done.".format(time.time() - tick)


def runner_SS(output_base, baseline_dir, baseline=False, analytical_mtrs=True,
              age_specific=False, reform=0, user_params={}, guid='',
              calibrate_model=False, run_micro=True):

    from ogusa import parameters, demographics, income, utils
    from ogusa import txfunc

    tick = time.time()

    #Create output directory structure
    saved_moments_dir = os.path.join(output_base, "Saved_moments")
    ss_dir = os.path.join(output_base, "SS")
    tpi_dir = os.path.join(output_base, "TPI")
    dirs = [saved_moments_dir, ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print "making dir: ", _dir
            os.makedirs(_dir)
        except OSError as oe:
            pass

    print ("in runner, baseline is ", baseline)
    run_params = ogusa.parameters.get_parameters(baseline=baseline, reform=reform,
                          guid=guid, user_modifiable=True)
    run_params['analytical_mtrs'] = analytical_mtrs



    # Modify ogusa parameters based on user input
    if 'frisch' in user_params:
        print "updating fricsh and associated"
        b_ellipse, upsilon = ogusa.elliptical_u_est.estimation(user_params['frisch'],
                                                               run_params['ltilde'])
        run_params['b_ellipse'] = b_ellipse
        run_params['upsilon'] = upsilon
        run_params.update(user_params)

    # Modify ogusa parameters based on user input
    if 'sigma' in user_params:
        print "updating sigma"
        run_params['sigma'] = user_params['sigma']
        run_params.update(user_params)

    from ogusa import SS, TPI


    # List of parameter names that will not be changing (unless we decide to
    # change them for a tax experiment)

    param_names = ['S', 'J', 'T', 'BW', 'lambdas', 'starting_age', 'ending_age',
                'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
                'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
                'analytical_mtrs', 'b_ellipse', 'k_ellipse', 'upsilon',
                'chi_b_guess', 'chi_n_guess','etr_params','mtrx_params',
                'mtry_params','tau_payroll', 'tau_bq',
                'retire', 'mean_income_data', 'g_n_vector',
                'h_wealth', 'p_wealth', 'm_wealth',
                'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'imm_rates', 'e', 'rho', 'omega_S_preTP']


    '''
    ------------------------------------------------------------------------
        If using income tax reform, need to determine parameters that yield
        same SS revenue as the wealth tax reform.
    ------------------------------------------------------------------------
    '''
    if reform == 1:
        sim_params = {}
        for key in param_names:
            sim_params[key] = run_params[key]

        sim_params['output_dir'] = output_base
        sim_params['run_params'] = run_params
        income_tax_params, ss_params, iterative_params, chi_params= SS.create_steady_state_parameters(**sim_params)

        # find SS revenue from wealth tax reform
        reform3_ss_dir = os.path.join(
        "./OUTPUT_WEALTH_REFORM"    + '/sigma' + str(run_params['sigma']), "SS/SS_vars.pkl")
        ss_solutions = pickle.load(open(reform3_ss_dir, "rb"))
        lump_to_match = ss_solutions['T_Hss']

        # create function to match SS revenue
        def matcher(d_guess, params):
            income_tax_params, lump_to_match = params
            analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
            etr_params[:,3] = d_guess
            mtrx_params[:,3] = d_guess
            mtry_params[:,3] = d_guess
            income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
            ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                              chi_params, baseline,baseline_dir=baseline_dir)

            lump_new = ss_solutions['T_Hss']
            error = abs(lump_to_match - lump_new)
            print 'Error in taxes:', error
            return error

        print 'Computing new income tax to match wealth tax'
        d_guess= .219 # initial guess
        import scipy.optimize as opt
        params = [income_tax_params, lump_to_match]
        new_d_inc = opt.fsolve(matcher, d_guess, args=params, xtol=1e-13)
        print '\tOld income tax:', d_guess
        print '\tNew income tax:', new_d_inc

        analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
        etr_params[:,:,3] = new_d_inc
        mtrx_params[:,:,3] = new_d_inc
        mtry_params[:,:,3] = new_d_inc
        income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
        ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                          chi_params, baseline,baseline_dir=baseline_dir)



    '''
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    '''

    sim_params = {}
    for key in param_names:
        sim_params[key] = run_params[key]

    sim_params['output_dir'] = output_base
    sim_params['run_params'] = run_params

    income_tax_params, ss_params, iterative_params, chi_params= SS.create_steady_state_parameters(**sim_params)

    '''
    ****
    CALL CALIBRATION here if boolean flagged
    ****
    '''
    if calibrate_model:
        chi_params = calibrate.chi_estimate(income_tax_params, ss_params,
                      iterative_params, chi_params, baseline_dir=baseline_dir)

    ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                      chi_params, baseline,baseline_dir=baseline_dir)

    '''
    ------------------------------------------------------------------------
        Pickle SS results
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
