'''
------------------------------------------------------------------------
Last updated 2/16/2014

Functions for created the matrix of ability levels, e.

This py-file calls the following other file(s):
            data/e_vec_data/cwhs_earn_rate_age_profile.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/ability_log
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt


'''
------------------------------------------------------------------------
    Generate Polynomials
------------------------------------------------------------------------

------------------------------------------------------------------------
'''

# Vals for: .25 .25 .2 .1 .1 .09 .01
one = np.array([-0.09720122, 0.05995294, 0.17654618, 0.21168263, 0.21638731, 0.04500235, 0.09229392])                 
two = np.array([0.00247639, -0.00004086, -0.00240656, -0.00306555, -0.00321041, 0.00094253, 0.00012902])                   
three = np.array([-0.00001842, -0.00000521, 0.00001039, 0.00001438, 0.00001579, -0.00001470, -0.00001169])             
constant = np.array([3.41e+00, 0.69689692, -0.78761958, -1.11e+00, -0.93939272, 1.60e+00, 1.89e+00])
ages = np.linspace(21, 80, 60)
ages = np.tile(ages.reshape(60, 1), (1, 7))
income_profiles = constant + one * ages + two * ages ** 2 + three * ages ** 3
income_profiles = np.exp(income_profiles)


'''
------------------------------------------------------------------------
    Generate ability type matrix
------------------------------------------------------------------------
Given desired starting and stopping ages, as well as the values for S
and J, the ability matrix is created.
------------------------------------------------------------------------
'''


def graph_income(S, J, e, starting_age, ending_age, bin_weights):
    '''
    Graphs the log of the ability matrix.
    '''
    domain = np.linspace(starting_age, ending_age, S)
    Jgrid = np.zeros(J)
    for j in xrange(J):
        Jgrid[j:] += bin_weights[j]
    X, Y = np.meshgrid(domain, Jgrid)
    cmap2 = matplotlib.cm.get_cmap('autumn')
    if J == 1:
        plt.figure()
        plt.plot(domain, np.log(e))
        plt.savefig('OUTPUT/Demographics/ability_log')
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, np.log(e).T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'log ability $log(e_j(s))$')
        plt.show()
        # plt.savefig('OUTPUT/Demographics/ability_log')
    if J == 1:
        plt.figure()
        plt.plot(domain, e)
        plt.savefig('OUTPUT/Demographics/ability')
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, e.T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'ability $e_j(s)$')
        # plt.savefig('OUTPUT/Demographics/ability')
        plt.show()


def arc_tan_func(points, a, b, c):
    y = (-a / np.pi) * np.arctan(b*points + c) + a / 2
    return y


def arc_tan_deriv_func(points, a, b, c):
    y = -a * b / (np.pi * (1+(b*points+c)**2))
    return y


def arc_error(guesses, params):
    a, b, c = guesses
    first_point, coef1, coef2, coef3 = params
    error1 = first_point - arc_tan_func(80, a, b, c)
    error2 = np.exp(3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) - arc_tan_deriv_func(80, a, b, c)
    error3 = .1 * first_point - arc_tan_func(95, a, b, c)
    error = [np.abs(error1)] + [np.abs(error2)] + [np.abs(error3)]
    print np.array(error).max()
    return error


def arc_tan_fit(first_point, coef1, coef2, coef3):
    guesses = [200, 2, 2]
    params = [first_point, coef1, coef2, coef3]
    a, b, c = opt.fsolve(arc_error, guesses, params)
    print a, b, c
    old_ages = np.linspace(81, 100, 20)
    return arc_tan_func(old_ages, a, b, c)


def get_e(S, J, starting_age, ending_age, bin_weights, omega_SS):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age
                starting_age - age of first age cohort
                ending_age - age of last age cohort
                bin_weights - what fraction of each age is in each
                              abiility type

    Returns:    e - S x J matrix of ability levels for each
                    age cohort, normalized so
                    the mean is one
    '''
    e_short = income_profiles
    e_final = np.ones((S, J))
    e_final[:60, :] = e_short
    e_final[60:, :] = 0.0
    for j in xrange(J):
        e_final[60:, j] = arc_tan_fit(e_final[59, j], one[j], two[j], three[j])
    graph_income(S, J, e_final, starting_age, ending_age, bin_weights)
    return e_final

get_e(80, 7, 21, 100, np.array([.25, .25, .2, .1, .1, .09, .01]), 0)
