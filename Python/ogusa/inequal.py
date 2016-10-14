'''
------------------------------------------------------------------------
Last updated 9/21/2016

Functions to compute various measures of inequality.

This python files calls:


------------------------------------------------------------------------
'''

# Packages
import os
from io import StringIO
import numpy as np
import cPickle as pickle
from pkg_resources import resource_stream, Requirement

EPSILON = 1e-10
PATH_EXISTS_ERRNO = 17

def the_inequalizer(dist, pop_weights, ability_weights, factor, S, J):
    '''
    --------------------------------------------------------------------
    Generates three measures of inequality.

    Inputs:
        dist            = [S,J] array, distribution of endogenous variables over age and lifetime income group
        pop_weights     = [S,] vector, fraction of population by each age
        ability_weights = [J,] vector, fraction of population for each lifetime income group
        factor          = scalar, factor relating model units to dollars
        S               = integer, number of economically active periods in lifetime
        J               = integer, number of ability types

    Functions called: None

    Objects in function:
        weights           = [S,J] array, fraction of population for each age and lifetime income group
        flattened_dist    = [S*J,] vector, vectorized dist
        flattened_weights = [S*J,] vector, vectorized weights
        sort_dist         = [S*J,] vector, ascending order vector of dist
        loc_90th          = integer, index of 90th percentile
        loc_10th          = integer, index of 10th percentile
        loc_99th          = integer, index of 99th percentile

    Returns: measure of inequality
    --------------------------------------------------------------------
    '''

    weights = np.tile(pop_weights.reshape(S, 1), (1, J)) * \
    ability_weights.reshape(1, J)
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)

    # gini
    p = np.cumsum(sort_weights)
    nu = np.cumsum(sort_dist*sort_weights)
    nu = nu/nu[-1]
    gini_coeff = (nu[1:]*p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    # variance
    ln_dist = np.log(sort_dist*factor) # not scale invariant
    weight_mean = (ln_dist*sort_weights).sum()/sort_weights.sum()
    var_ln_dist = ((sort_weights*((ln_dist-weight_mean)**2)).sum())*(1./(sort_weights.sum()))


    # 90/10 ratio
    loc_90th = np.argmin(np.abs(cum_weights - .9))
    loc_10th = np.argmin(np.abs(cum_weights - .1))
    ratio_90_10 = sort_dist[loc_90th] / sort_dist[loc_10th]

    # top 10% share
    top_10_share= (sort_dist[loc_90th:] * sort_weights[loc_90th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # top 1% share
    loc_99th = np.argmin(np.abs(cum_weights - .99))
    top_1_share = (sort_dist[loc_99th:] * sort_weights[loc_99th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # calculate percentile shares (percentiles based on lambdas input)
    dist_weight = (sort_weights*sort_dist)
    total_dist_weight = dist_weight.sum()
    cumsum = np.cumsum(sort_weights)
    dist_sum = np.zeros((J,))
    cum_weights = ability_weights.cumsum()
    for i in range(J):
        cutoff = sort_weights.sum() / (1./cum_weights[i])
        dist_sum[i] = ((dist_weight[cumsum < cutoff].sum())/total_dist_weight)


    dist_share = np.zeros((J,))
    dist_share[0] = dist_sum[0]
    dist_share[1:] = dist_sum[1:]-dist_sum[0:-1]

    return np.append([dist_share], [gini_coeff,var_ln_dist])

def gini(dist, weights):
    '''
    --------------------------------------------------------------------
    Calculates the gini coefficient.
    --------------------------------------------------------------------
    '''

    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]/flattened_weights.sum()
    p = np.cumsum(sort_weights)

    # gini
    nu = np.cumsum(sort_dist*sort_weights)
    nu = nu/nu[-1]
    gini_coeff = (nu[1:]*p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    return gini_coeff

def ninety_ten(dist, weights):
    '''
    --------------------------------------------------------------------
    Calculates ratio of the 90th to 10th percentile.
    --------------------------------------------------------------------
    '''
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)


    # 90/10 ratio
    loc_90th = np.argmin(np.abs(cum_weights - .9))
    loc_10th = np.argmin(np.abs(cum_weights - .1))
    ratio_90_10 = sort_dist[loc_90th] / sort_dist[loc_10th]

    return ratio_90_10


def top_1(dist, weights):
    '''
    --------------------------------------------------------------------
    Calculates the top 1% share
    --------------------------------------------------------------------
    '''
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)


    # top 1% share
    loc_99th = np.argmin(np.abs(cum_weights - .99))
    top_1_share = (sort_dist[loc_99th:] * sort_weights[loc_99th:]
           ).sum() / (sort_dist * sort_weights).sum()

    return top_1_share


def top_10(dist, weights):
    '''
    --------------------------------------------------------------------
    Calculates the top 10% share
    --------------------------------------------------------------------
    '''
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)


    # top 10% share
    loc_90th = np.argmin(np.abs(cum_weights - .9))
    top_10_share= (sort_dist[loc_90th:] * sort_weights[loc_90th:]
           ).sum() / (sort_dist * sort_weights).sum()

    return top_10_share


def var_log(dist, weights,factor):
    '''
    --------------------------------------------------------------------
    Calculates the variance in logs
    --------------------------------------------------------------------
    '''
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]

    # variance
    ln_dist = np.log(sort_dist*factor) # not scale invariant
    weight_mean = (ln_dist*sort_weights).sum()/sort_weights.sum()
    var_ln_dist = ((sort_weights*((ln_dist-weight_mean)**2)).sum())*(1./(sort_weights.sum()))

    return var_ln_dist
