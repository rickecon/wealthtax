''' 
------------------------------------------------------------------------
Last updated 03/04/2015

Import Coefficients from wage profile regression using
IRS CWHS sample.  Plot the profiles for each lifetime
income group..

This py-file calls the following other file(s):
            None

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

'''
------------------------------------------------------------------------
------------------------------------------------------------------------
------------------------------------------------------------------------
'''

# Working directory and input file name
path = '/Users/jasondebacker/repos/wealthtax/Data/IncomeProcesses/'
file_name = 'wage_profiles.csv' ;

# read in the csv file with the regression coefficients
wage_df = pd.read_csv(path + file_name)

plt.figure(); wage_df.plot(); plt.legend(loc='best')
savefig(path+'wage_profiles.png')
