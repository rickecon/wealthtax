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

## Labor dist from data with extrapolation

## Calibrated values of chi_n

## Wealth over the lifecycle, model vs data for 7 percentile groups

## SS distribution of labor supply, baseline, by ability type

## SS distribution of consumption, baseline, by ability type

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
