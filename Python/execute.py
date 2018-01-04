'''
A 'smoke test' for the ogusa package. Uses a fake data set to run the
baseline
'''

import cPickle as pickle
import os
import numpy as np
import time

import ogusa
from ogusa import calibrate, wealth
ogusa.parameters.DATASET = 'REAL'


def runner(output_base, baseline_dir, baseline=False, analytical_mtrs=True,
           age_specific=False, reform=0, fix_transfers=False, user_params={}, guid='',
           run_micro=True, calibrate_model=False):

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
        reform3_ss_solutions = pickle.load(open(reform3_ss_dir, "rb"))
        receipts_to_match = reform3_ss_solutions['T_Hss']

        # create function to match SS revenue
        # def matcher(d_guess, params):
        #     income_tax_params, receipts_to_match, ss_params, iterative_params,\
        #                       chi_params, baseline, baseline_dir = params
        #     analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
        #     etr_params[:,3] = d_guess
        #     mtrx_params[:,3] = d_guess
        #     mtry_params[:,3] = d_guess
        #     income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
        #     ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
        #                       chi_params, baseline ,baseline_dir=baseline_dir, output_base=output_base)
        #
        #     receipts_new = ss_outputs['T_Hss'] + ss_outputs['Gss']
        #     error = abs(receipts_to_match - receipts_new)
        #     if d_guess <= 0:
        #         error = 1e14
        #     print 'Error in taxes:', error
        #     return error

        # print 'Computing new income tax to match wealth tax'
        d_guess= 0.503 # initial guess
        # import scipy.optimize as opt
        # params = [income_tax_params, receipts_to_match, ss_params, iterative_params,
        #                   chi_params, baseline, baseline_dir]
        # new_d_inc = opt.fsolve(matcher, d_guess, args=params, xtol=1e-6)
        new_d_inc = 0.503  # this value comes out given default parameter values (if fix_transfers=True this is 0.503 if False then 0.453)

        print '\tOld income tax:', d_guess
        print '\tNew income tax:', new_d_inc
        analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

        etr_params[:,3] = new_d_inc
        mtrx_params[:,3] = new_d_inc
        mtry_params[:,3] = new_d_inc

        run_params['etr_params'] = np.tile(np.reshape(etr_params,(run_params['S'],1, etr_params.shape[1])),(1,run_params['BW'],1))
        run_params['mtrx_params'] = np.tile(np.reshape(mtrx_params,(run_params['S'],1,mtrx_params.shape[1])),(1,run_params['BW'],1))
        run_params['mtry_params'] = np.tile(np.reshape(mtry_params,(run_params['S'],1,mtry_params.shape[1])),(1,run_params['BW'],1))

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

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    print('ETR param shape = ', etr_params.shape)

    ss_outputs = SS.run_SS(income_tax_params, ss_parameters, iterative_params,
                           chi_params, baseline, fix_transfers=fix_transfers,
                           baseline_dir=baseline_dir)

    '''
    ------------------------------------------------------------------------
        Pickle SS results and parameters of run
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        param_dir = os.path.join(baseline_dir, "run_parameters.pkl")
        pickle.dump(sim_params, open(param_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        param_dir = os.path.join(output_base, "run_parameters.pkl")
        pickle.dump(sim_params, open(param_dir, "wb"))


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
        tpi_params, iterative_params, initial_values, SS_values,
        fix_transfers=fix_transfers, output_dir=output_base)


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
              age_specific=False, reform=0, fix_transfers=False, user_params={}, guid='',
              calibrate_model=False, run_micro=True):

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

    from ogusa import SS, TPI, SS_alt


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


    sim_params = {}
    for key in param_names:
        sim_params[key] = run_params[key]

    sim_params['output_dir'] = output_base
    sim_params['run_params'] = run_params

    '''
    ------------------------------------------------------------------------
        If using income tax reform, need to determine parameters that yield
        same SS revenue as the wealth tax reform.
    ------------------------------------------------------------------------
    '''
    if reform == 1:

        income_tax_params, ss_params, iterative_params, chi_params= SS.create_steady_state_parameters(**sim_params)

        # find SS revenue from wealth tax reform
        reform3_ss_dir = os.path.join(
        "./OUTPUT_WEALTH_REFORM"    + '/sigma' + str(run_params['sigma']), "SS/SS_vars.pkl")
        reform3_ss_solutions = pickle.load(open(reform3_ss_dir, "rb"))
        receipts_to_match = reform3_ss_solutions['T_Hss'] + reform3_ss_solutions['Gss']

        # create function to match SS revenue
        def matcher(d_guess, params):
            income_tax_params, receipts_to_match, ss_params, iterative_params,\
                              chi_params, baseline, baseline_dir = params
            analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
            etr_params[:,3] = d_guess
            mtrx_params[:,3] = d_guess
            mtry_params[:,3] = d_guess
            income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
            ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                              chi_params, baseline, fix_transfers=fix_transfers,
                              baseline_dir=baseline_dir)

            receipts_new = ss_outputs['T_Hss'] + ss_outputs['Gss']
            error = abs(receipts_to_match - receipts_new)
            if d_guess <= 0:
                error = 1e14
            print 'Error in taxes:', error
            return error

        print 'Computing new income tax to match wealth tax'
        # d_guess= .452 # initial guess 0.452 works for sigma = 2, frisch 1.5
        # new_d_inc = d_guess
        # import scipy.optimize as opt
        # params = [income_tax_params, receipts_to_match, ss_params, iterative_params,
        #                   chi_params, baseline, baseline_dir]
        # new_d_inc = opt.fsolve(matcher, d_guess, args=params, xtol=1e-8)
        # print '\tOld income tax:', d_guess
        # print '\tNew income tax:', new_d_inc

        def samesign(a, b):
            return a * b > 0

        def bisect_method(func, params, low, high):
            'Find root of continuous function where f(low) and f(high) have opposite signs'

            #assert not samesign(func(params,low), func(params,high))

            for i in range(54):
                midpoint = (low + high) / 2.0
                if samesign(func(params,low), func(params,midpoint)):
                    low = midpoint
                else:
                    high = midpoint

            return midpoint

        def solve_model(params,d):
            income_tax_params, ss_params, iterative_params,\
                              chi_params, baseline ,baseline_dir = params
            analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
            etr_params[:,3] = d
            mtrx_params[:,3] = d
            mtry_params[:,3] = d
            income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
            ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                              chi_params, baseline, fix_transfers=fix_transfers,
                              baseline_dir=baseline_dir)
            ss_dir = os.path.join("./OUTPUT_INCOME_REFORM/sigma2.0", "SS/SS_vars.pkl")
            pickle.dump(ss_outputs, open(ss_dir, "wb"))
            receipts_new = ss_outputs['T_Hss'] + ss_outputs['Gss']
            new_error = receipts_to_match - receipts_new
            print 'Error in taxes:', error
            print 'New income tax:', d
            return new_error

        # print 'Computing new income tax to match wealth tax'
        # d_guess= 0.5025 # initial guess
        # # income_tax_params, receipts_to_match, ss_params, iterative_params,\
        # #                   chi_params, baseline, baseline_dir = params
        # analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
        # etr_params[:,3] = d_guess
        # mtrx_params[:,3] = d_guess
        # mtry_params[:,3] = d_guess
        # income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
        # ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
        #                   chi_params, baseline, fix_transfers=fix_transfers,
        #                   baseline_dir=baseline_dir)
        # ss_dir = os.path.join("./OUTPUT_INCOME_REFORM/sigma2.0", "SS/SS_vars.pkl")
        # pickle.dump(ss_outputs, open(ss_dir, "wb"))
        # receipts_new = ss_outputs['T_Hss'] + ss_outputs['Gss']
        # error = receipts_to_match - receipts_new
        # new_error = error
        # print "ERROR: ", error
        # max_loop_iter = 1
        # output_list = np.zeros((max_loop_iter,3))
        # loop_iter = 0
        # bisect = 0
        # d_guess_old = d_guess
        # # while np.abs(new_error) > 1e-8 and loop_iter < max_loop_iter:
        # while loop_iter < max_loop_iter:
        #     # if new_error > 0 and new_error > 0 and bisect == 0:
        #     #     d_guess_old = d_guess
        #     #     d_guess+=0.001
        #     # elif new_error < 0 and new_error < 0 and bisect == 0:
        #     #     d_guess_old = d_guess
        #     #     d_guess-=0.001
        #     #     d_guess = max(0.0,d_guess) # constrain so not negative
        #     # else:
        #     #     bisect = 1
        #     #     print 'Entering bisection method'
        #     #     params = income_tax_params, ss_params, iterative_params,\
        #     #                       chi_params, baseline ,baseline_dir
        #     #     high = max(d_guess,d_guess_old)
        #     #     low = min(d_guess,d_guess_old)
        #     #     d_guess = bisect_method(solve_model, params, low, high)
        #     #     loop_iter = max_loop_iter
        #     d_guess_old = d_guess
        #     d_guess+=0.0005
        #
        #     error = new_error
        #     etr_params[:,3] = d_guess
        #     mtrx_params[:,3] = d_guess
        #     mtry_params[:,3] = d_guess
        #     income_tax_params = analytical_mtrs, etr_params, mtrx_params, mtry_params
        #     print 'now here$$$'
        #     ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
        #                       chi_params, baseline, fix_transfers=fix_transfers,
        #                       baseline_dir=baseline_dir)
        #     ss_dir = os.path.join("./OUTPUT_INCOME_REFORM/sigma2.0", "SS/SS_vars.pkl")
        #     pickle.dump(ss_outputs, open(ss_dir, "wb"))
        #     receipts_new = ss_outputs['T_Hss'] + ss_outputs['Gss']
        #     new_error = (receipts_to_match - receipts_new)
        #     print "ERROR: ", new_error
        #     output_list[loop_iter,0]=new_error
        #     output_list[loop_iter,1]=d_guess
        #     output_list[loop_iter,2]=ss_outputs['Yss']-ss_outputs['Iss']-ss_outputs['Css']-ss_outputs['Gss']
        #     np.savetxt('inc_tax_out.csv',output_list, delimiter=",")
        #     pickle.dump(output_list, open("output_list.pkl", "wb"))
        #     print 'Error in taxes:', error
        #     print 'Old income tax:', d_guess_old
        #     print 'New income tax:', d_guess
        #     print 'iteration: ', loop_iter
        #     loop_iter += 1

        analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
        new_d_inc = 0.5025 # this is 0.453 if fix_transfers=False, 0.503 if True
        etr_params[:,3] = new_d_inc
        mtrx_params[:,3] = new_d_inc
        mtry_params[:,3] = new_d_inc

        sim_params['etr_params'] = np.tile(np.reshape(etr_params,(run_params['S'],1,etr_params.shape[1])),(1,run_params['BW'],1))
        sim_params['mtrx_params'] = np.tile(np.reshape(mtrx_params,(run_params['S'],1,mtrx_params.shape[1])),(1,run_params['BW'],1))
        sim_params['mtry_params'] = np.tile(np.reshape(mtry_params,(run_params['S'],1,mtry_params.shape[1])),(1,run_params['BW'],1))


    '''
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    '''
    income_tax_params, ss_params, iterative_params, chi_params= SS.create_steady_state_parameters(**sim_params)
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    '''
    ****
    CALL CALIBRATION here if boolean flagged
    ****
    '''
    if calibrate_model:
        chi_params = calibrate.chi_estimate(income_tax_params, ss_params,
                      iterative_params, chi_params, baseline_dir=baseline_dir)

    # ss_outputs = SS_alt.run_SS(income_tax_params, ss_params, iterative_params,
    #                   chi_params, baseline, baseline_dir=baseline_dir)
    print 'Fix transfers = ', fix_transfers
    ss_outputs = SS.run_SS(income_tax_params, ss_params, iterative_params,
                      chi_params, baseline, fix_transfers=fix_transfers,
                      baseline_dir=baseline_dir)



    model_moments = ogusa.calibrate.calc_moments(ss_outputs,
                                                 sim_params['omega_SS'],
                                                 sim_params['lambdas'],
                                                 sim_params['S'],
                                                 sim_params['J'])

    scf, data = ogusa.wealth.get_wealth_data()
    wealth_moments = ogusa.wealth.compute_wealth_moments(scf, sim_params['lambdas'], sim_params['J'])

    print 'model moments: ', model_moments[:sim_params['J']+2]
    print 'data moments: ', wealth_moments

    '''
    ------------------------------------------------------------------------
        Pickle SS results and parameters
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        param_dir = os.path.join(baseline_dir, "run_parameters.pkl")
        pickle.dump(sim_params, open(param_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        param_dir = os.path.join(output_base, "run_parameters.pkl")
        pickle.dump(sim_params, open(param_dir, "wb"))
