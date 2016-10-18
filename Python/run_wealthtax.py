import ogusa
import os
import sys
from multiprocessing import Process
import time

#import postprocess
#from execute import runner # change here for small jobs
from execute import runner, runner_SS


def run_micro_macro(user_params):

    start_time = time.time()

    baseline_dir = "./OUTPUT_BASELINE"

    '''
    ------------------------------------------------------------------------
        Run calibration
    -----------------------------------------------------------------------
    '''
    output_base = baseline_dir
    input_dir = baseline_dir
    user_params = {'frisch':(1/1.5), 'sigma':3.0}
    kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
            'baseline':True, 'reform':0, 'user_params':user_params,
            'guid':'wealth_tax_baseline','calibrate_model':False}
    runner_SS(**kwargs)

    sigma_list = [3.0, 1.1, 2.1, 3.2]

    '''
    Loop over value of sigma and run all baselines and reforms
    '''
    for item in sigma_list:
        print 'item= ', item
        # parameters that may update at each iteration
        user_params = {'frisch':(1/1.5), 'sigma':item}

        # set up directories to save output to
        baseline_dir = "./OUTPUT_BASELINE" + '/sigma' + str(item)
        wealth_dir = "./OUTPUT_WEALTH_REFORM" + '/sigma' + str(item)
        income_dir = "./OUTPUT_INCOME_REFORM" + '/sigma' + str(item)


        '''
        ------------------------------------------------------------------------
            Run SS for Baseline first
        ------------------------------------------------------------------------
        '''
        output_base = baseline_dir
        input_dir = baseline_dir
        kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
                'baseline':True, 'reform':0, 'user_params':user_params,
                'guid':'baseline_sigma_'+str(item),'calibrate_model':False}
        runner_SS(**kwargs)


        '''
        ------------------------------------------------------------------------
            Run baseline TPI
        ------------------------------------------------------------------------
        '''
        # output_base = baseline_dir
        # input_dir = baseline_dir
        # kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
        #         'baseline':True,'reform':0,'user_params':user_params,
        #         'guid':'baseline_sigma_'+str(item),'calibrate_model':False}
        # runner(**kwargs)

        '''
        ------------------------------------------------------------------------
            Run wealth tax reform (needs to be run before income tax reform
            because it determines the SS revenue target)
        ------------------------------------------------------------------------
        '''
        output_base = wealth_dir
        input_dir = wealth_dir
        guid_iter = 'reform_' + str(0)
        kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
                'baseline':False, 'reform':2, 'user_params':user_params,
                'guid':'wealth_tax_reform2','calibrate_model':False}
        runner_SS(**kwargs)

        '''
        ------------------------------------------------------------------------
            Run income tax reform
        ------------------------------------------------------------------------
        '''
        output_base = income_dir
        input_dir = income_dir
        guid_iter = 'reform_' + str(0)
        kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
                'baseline':False, 'reform':1, 'user_params':user_params,
                'guid':'wealth_tax_reform1','calibrate_model':False}
        runner_SS(**kwargs)



    # run post process script to create tables/figures for paper
    #postprocess('directories')

if __name__ == "__main__":
    run_micro_macro(user_params={})
