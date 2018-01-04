'''
------------------------------------------------------------------------
Last updated 4/7/2016

Household functions for taxes in the steady state and along the
transition path..

This file calls the following files:
    tax.py
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import tax

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_K(b, params):
    '''
    Calculates aggregate capital supplied.

    Inputs:
        b           = [T,S,J] array, distribution of wealth/capital holdings
        params      = length 4 tuple, (omega, lambdas, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        g_n         = [T,] vector, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        K_presum = [T,S,J] array, weighted distribution of wealth/capital holdings
        K        = [T,] vector, aggregate capital supply

    Returns: K
    '''

    omega, lambdas, imm_rates, g_n, method = params

    if method == 'SS':
        part1 = b* omega * lambdas
        omega_extended = np.append(omega[1:],[0.0])
        imm_extended = np.append(imm_rates[1:],[0.0])
        part2 = b*(omega_extended*imm_extended).reshape(omega.shape[0],1)*lambdas
        K_presum = part1+part2
        K = K_presum.sum()
    elif method == 'TPI':
        part1 = b* omega * lambdas
        #omega_extended = np.append(omega[1:,:,:],np.zeros((1,omega.shape[1],omega.shape[2])),axis=0)
        omega_shift = np.append(omega[:,1:,:],np.zeros((omega.shape[0],1,omega.shape[2])),axis=1)
        #imm_extended = np.append(imm_rates[1:,:,:],np.zeros((1,imm_rates.shape[1],imm_rates.shape[2])),axis=0)
        imm_shift = np.append(imm_rates[:,1:,:],np.zeros((imm_rates.shape[0],1,imm_rates.shape[2])),axis=1)
        #part2 = b*(omega_extended*imm_extended)*lambdas
        part2 = b*imm_shift*omega_shift*lambdas
        K_presum = part1+part2
        K = K_presum.sum(1).sum(1)
    K /= (1.0 + g_n)
    return K


def get_BQ(r, b_splus1, params):
    '''
    Calculation of bequests to each lifetime income group.

    Inputs:
        r           = [T,] vector, interest rates
        b_splus1    = [T,S,J] array, distribution of wealth/capital holdings one period ahead
        params      = length 5 tuple, (omega, lambdas, rho, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        rho         = [S,] vector, mortality rates
        g_n         = scalar, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        BQ_presum = [T,S,J] array, weighted distribution of wealth/capital holdings one period ahead
        BQ        = [T,J] array, aggregate bequests by lifetime income group

    Returns: BQ
    '''
    omega, lambdas, rho, g_n, method = params

    BQ_presum = b_splus1 * omega * rho * lambdas
    if method == 'SS':
        BQ = BQ_presum.sum(0)
    elif method == 'TPI':
        BQ = BQ_presum.sum(1)
    BQ *= (1.0 + r) / (1.0 + g_n)
    return BQ


def marg_ut_cons(cvec, sigma):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.0001
    if np.ndim(cvec) == 0:
        c_s = cvec
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(cvec) == 1:
        p = cvec.shape[0]
        cvec_cnstr = cvec < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = cvec[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * cvec[cvec_cnstr] + b1

    return MU_c


def marg_ut_labor(nvec, params):
    '''
    --------------------------------------------------------------------
    Generate marginal disutility(ies) of labor with elliptical
    disutility of labor function and stitched functions at lower bound
    and upper bound of labor supply such that the new hybrid function is
    defined over all labor supply on the real line but the function has
    similar properties to the Inada conditions at the upper and lower
    bounds.

    v'(n) = (b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n / l_tilde) ** upsilon)) ** ((1-upsilon)/upsilon))
            if n >= eps_low <= n <= eps_high
          = g_low'(n)  = 2 * b2 * n + b1 if n < eps_low
          = g_high'(n) = 2 * d2 * n + d1 if n > eps_high

        such that g_low'(eps_low) = u'(eps_low)
        and g_low''(eps_low) = u''(eps_low)
        and g_high'(eps_high) = u'(eps_high)
        and g_high''(eps_high) = u''(eps_high)

        v(n) = -b *(1 - ((n/l_tilde) ** upsilon)) ** (1/upsilon)
        g_low(n)  = b2 * (n ** 2) + b1 * n + b0
        g_high(n) = d2 * (n ** 2) + d1 * n + d0
    --------------------------------------------------------------------
    INPUTS:
    nvec   = scalar or (p,) vector, labor supply value or labor supply
             values over remaining periods of lifetime
    params = length 3 tuple, (l_tilde, b_ellip, upsilon)
    graph  = Boolean, =True if want plot of stitched marginal disutility
             of labor function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    l_tilde       = scalar > 0, time endowment for each agent each per
    b_ellip       = scalar > 0, scale parameter for elliptical utility
                    of leisure function
    upsilon       = scalar > 1, shape parameter for elliptical utility
                    of leisure function
    eps_low       = scalar > 0, positive value close to zero
    eps_high      = scalar > 0, positive value just less than l_tilde
    n_s           = scalar, individual labor supply
    n_s_low       = boolean, =True for n_s < eps_low
    n_s_high      = boolean, =True for n_s > eps_high
    n_s_uncstr    = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high
    MDU_n         = scalar or (p,) vector, marginal disutility or
                    marginal utilities of labor supply
    b1            = scalar, intercept value in linear marginal
                    disutility of labor at lower bound
    b2            = scalar, slope coefficient in linear marginal
                    disutility of labor at lower bound
    d1            = scalar, intercept value in linear marginal
                    disutility of labor at upper bound
    d2            = scalar, slope coefficient in linear marginal
                    disutility of labor at upper bound
    p             = integer >= 1, number of periods remaining in life
    nvec_s_low    = boolean, =True for n_s < eps_low
    nvec_s_high   = boolean, =True for n_s > eps_high
    nvec_s_uncstr = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high

    FILES CREATED BY THIS FUNCTION:
        MDU_n_stitched.png

    RETURNS: MDU_n
    --------------------------------------------------------------------
    '''
    b_ellip, upsilon, l_tilde, chi_n = params

    eps_low = 0.000001
    eps_high = l_tilde - 0.000001
    # This if is for when nvec is a scalar
    if np.ndim(nvec) == 0:
        n_s = nvec
        n_s_low = n_s < eps_low
        n_s_high = n_s > eps_high
        n_s_uncstr = (n_s >= eps_low) and (n_s <= eps_high)
        if n_s_uncstr:
            MDU_n = \
                ((b_ellip / l_tilde) * ((n_s / l_tilde) **
                 (upsilon - 1)) * ((1 - ((n_s / l_tilde) ** upsilon)) **
                 ((1 - upsilon) / upsilon)))
        elif n_s_low:
            b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_low ** (upsilon - 2)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_low / l_tilde) ** upsilon) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
            b1 = ((b_ellip / l_tilde) * ((eps_low / l_tilde) **
                  (upsilon - 1)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
            MDU_n = 2 * b2 * n_s + b1
        elif n_s_high:
            d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_high ** (upsilon - 2)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_high / l_tilde) ** upsilon) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
            d1 = ((b_ellip / l_tilde) * ((eps_high / l_tilde) **
                  (upsilon - 1)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
            MDU_n = 2 * d2 * n_s + d1
    # This if is for when nvec is a one-dimensional vector
    elif np.ndim(nvec) == 1:
        p = nvec.shape[0]
        nvec_low = nvec < eps_low
        nvec_high = nvec > eps_high
        nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
        MDU_n = np.zeros(p)
        MDU_n[nvec_uncstr] = (
            (b_ellip / l_tilde) *
            ((nvec[nvec_uncstr] / l_tilde) ** (upsilon - 1)) *
            ((1 - ((nvec[nvec_uncstr] / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_low ** (upsilon - 2)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) *
              (1 + ((eps_low / l_tilde) ** upsilon) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
        b1 = ((b_ellip / l_tilde) * ((eps_low / l_tilde) **
              (upsilon - 1)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
        MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
        d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_high ** (upsilon - 2)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) *
              (1 + ((eps_high / l_tilde) ** upsilon) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
        d1 = ((b_ellip / l_tilde) * ((eps_high / l_tilde) **
              (upsilon - 1)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
        MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1

    output = chi_n * MDU_n

    return output


def get_cons(r, w, b, b_splus1, n, BQ, net_tax, params):
    '''
    Calculation of househld consumption.

    Inputs:
        r        = [T,] vector, interest rates
        w        = [T,] vector, wage rates
        b        = [T,S,J] array, distribution of wealth/capital holdings
        b_splus1 = [T,S,J] array, distribution of wealth/capital holdings one period ahead
        n        = [T,S,J] array, distribution of labor supply
        BQ       = [T,J] array, bequests by lifetime income group
        net_tax  = [T,S,J] array, distribution of net taxes
        params    = length 3 tuple (e, lambdas, g_y)
        e        = [S,J] array, effective labor units by age and lifetime income group
        lambdas  = [S,] vector, fraction of population in each lifetime income group
        g_y      = scalar, exogenous labor augmenting technological growth

    Functions called: None

    Objects in function:
        cons = [T,S,J] array, household consumption

    Returns: cons
    '''
    e, lambdas, g_y = params

    cons = (1 + r) * b + w * e * n + BQ / \
        lambdas - b_splus1 * np.exp(g_y) - net_tax
    return cons


def get_C(c, params):
    '''
    Calculation of aggregate consumption.

    Inputs:
        cons        = [T,S,J] array, household consumption
        params      = length 3 tuple (omega, lambdas, method)
        omega       = [S,T] array, population weights by age (Sx1 array)
        lambdas     = [J,1] vector, lifetime income group weights
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        aggC_presum = [T,S,J] array, weighted consumption by household
        aggC        = [T,] vector, aggregate consumption

    Returns: aggC
    '''

    omega, lambdas, method = params

    aggC_presum = c * omega * lambdas
    if method == 'SS':
        aggC = aggC_presum.sum()
    elif method == 'TPI':
        aggC = aggC_presum.sum(1).sum(1)
    return aggC


def FOC_savings(r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H, params):
    '''
    Computes Euler errors for the FOC for savings in the steady state.
    This function is usually looped through over J, so it does one lifetime income group at a time.

    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital holdings
        b_splus1    = [S,J] array, distribution of wealth/capital holdings one period ahead
        b_splus2    = [S,J] array, distribution of wealth/capital holdings two periods ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime income group
        factor      = scalar, scaling factor to convert model income to dollars
        T_H         = scalar, lump sum transfer
        params      = length 18 tuple (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas,
                                    J, S, etr_params, mtry_params, h_wealth, p_wealth,
                                    m_wealth, tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        beta        = scalar, discount factor
        g_y         = scalar, exogenous labor augmenting technological growth
        chi_b       = [J,] vector, utility weight on bequests for each lifetime income group
        theta       = [J,] vector, replacement rate for each lifetime income group
        tau_bq      = scalar, bequest tax rate (scalar)
        rho         = [S,] vector, mortality rates
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in lifetime
        etr_params  = [S,10] array, parameters of effective income tax rate function
        mtry_params = [S,10] array, parameters of marginal tax rate on capital income function
        h_wealth    = scalar, parameter in wealth tax function
        p_wealth    = scalar, parameter in wealth tax function
        m_wealth    = scalar, parameter in wealth tax function
        tau_payroll = scalar, payroll tax rate
        tau_bq      = scalar, bequest tax rate

    Functions called:
        get_cons
        marg_ut_cons
        tax.total_taxes
        tax.MTR_capital

    Objects in function:
        tax1 = [S,J] array, net taxes in the current period
        tax2 = [S,J] array, net taxes one period ahead
        cons1 = [S,J] array, consumption in the current period
        cons2 = [S,J] array, consumption one period ahead
        deriv = [S,J] array, after-tax return on capital
        savings_ut = [S,J] array, marginal utility from savings
        euler = [S,J] array, Euler error from FOC for savings

    Returns: euler
    '''
    e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, J, S, \
        analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, method = params

    # In order to not have 2 savings euler equations (one that solves the first S-1 equations, and one that solves the last one),
    # we combine them.  In order to do this, we have to compute a consumption term in period t+1, which requires us to have a shifted
    # e and n matrix.  We append a zero on the end of both of these so they will be the right size.  We could append any value to them,
    # since in the euler equation, the coefficient on the marginal utility of
    # consumption for this term will be zero (since rho is one).
    if method == 'TPI_scalar':
        e_extended = np.array([e] + [0])
        n_extended = np.array([n] + [0])
        etr_params_to_use = etr_params
        mtry_params_to_use = mtry_params
    else:
        e_extended = np.array(list(e) + [0])
        n_extended = np.array(list(n) + [0])
        etr_params_to_use = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
        mtry_params_to_use = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]

    # tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    # tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    # tax2_params = (e_extended[1:], lambdas, method, retire,
    #                np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:],
    #                h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    # tax2 = tax.total_taxes(r, w, b_splus1, n_extended[1:], BQ, factor, T_H, None, True, tax2_params)
    # cons1_params = (e, lambdas, g_y)
    # cons1 = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons1_params)
    # cons2_params = (e_extended[1:], lambdas, g_y)
    # cons2 = get_cons(r, w, b_splus1, b_splus2, n_extended[1:], BQ, tax2, cons2_params)

    # mtr_cap_params = (e_extended[1:], np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:],
    #                   np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:],analytical_mtrs)
    # deriv = (1+r) - r*(tax.MTR_capital(r, w, b_splus1, n_extended[1:], factor, mtr_cap_params))

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    tax2_params = (e_extended[1:], lambdas, method, retire,
                   etr_params_to_use, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax2 = tax.total_taxes(r, w, b_splus1, n_extended[1:], BQ, factor, T_H, None, True, tax2_params)
    cons1_params = (e, lambdas, g_y)
    cons1 = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons1_params)
    cons2_params = (e_extended[1:], lambdas, g_y)
    cons2 = get_cons(r, w, b_splus1, b_splus2, n_extended[1:], BQ, tax2, cons2_params)

    mtr_cap_params = (e_extended[1:], etr_params_to_use,
                      mtry_params_to_use,analytical_mtrs)
    deriv = (1+r) - r*(tax.MTR_capital(r, w, b_splus1, n_extended[1:], factor, mtr_cap_params))

    savings_ut = rho * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)

    # Again, note timing in this equation, the (1-rho) term will zero out in the last period, so the last entry of cons2 can be complete
    # gibberish (which it is).  It just has to exist so cons2 is the right
    # size to match all other arrays in the equation.
    euler = marg_ut_cons(cons1, sigma) - beta * (1 - rho) * deriv * marg_ut_cons(
        cons2, sigma) * np.exp(-sigma * g_y) - savings_ut


    return euler


def FOC_labor(r, w, b, b_splus1, n, BQ, factor, T_H, params):
    '''
    Computes Euler errors for the FOC for labor supply in the steady state.
    This function is usually looped through over J, so it does one lifetime income group at a time.

    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital holdings
        b_splus1    = [S,J] array, distribution of wealth/capital holdings one period ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime income group
        factor      = scalar, scaling factor to convert model income to dollars
        T_H         = scalar, lump sum transfer
        params      = length 19 tuple (e, sigma, g_y, theta, b_ellipse, upsilon, ltilde,
                                    chi_n, tau_bq, lambdas, J, S,
                                    etr_params, mtrx_params, h_wealth, p_wealth,
                                    m_wealth, tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        g_y         = scalar, exogenous labor augmenting technological growth
        theta       = [J,] vector, replacement rate for each lifetime income group
        b_ellipse   = scalar, scaling parameter in elliptical utility function
        upsilon     = curvature parameter in elliptical utility function
        chi_n       = [S,] vector, utility weights on disutility of labor
        ltilde      = scalar, upper bound of household labor supply
        tau_bq      = scalar, bequest tax rate (scalar)
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in lifetime
        etr_params  = [S,10] array, parameters of effective income tax rate function
        mtrx_params = [S,10] array, parameters of marginal tax rate on labor income function
        h_wealth    = scalar, parameter in wealth tax function
        p_wealth    = scalar, parameter in wealth tax function
        m_wealth    = scalar, parameter in wealth tax function
        tau_payroll = scalar, payroll tax rate
        tau_bq      = scalar, bequest tax rate

    Functions called:
        get_cons
        marg_ut_cons
        marg_ut_labor
        tax.total_taxes
        tax.MTR_labor

    Objects in function:
        tax = [S,J] array, net taxes in the current period
        cons = [S,J] array, consumption in the current period
        deriv = [S,J] array, net of tax share of labor income
        euler = [S,J] array, Euler error from FOC for labor supply

    Returns: euler
    '''
    e, sigma, g_y, theta, b_ellipse, upsilon, chi_n, ltilde, tau_bq, lambdas, J, S, \
        analytical_mtrs, etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, method  = params

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth,
                  m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    cons_params = (e, lambdas, g_y)
    cons = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons_params)
    mtr_lab_params = (e, etr_params, mtrx_params, analytical_mtrs)
    deriv = (1 - tau_payroll - tax.MTR_labor(r, w, b, n, factor, mtr_lab_params))

    lab_params = (b_ellipse, upsilon, ltilde, chi_n)
    euler = marg_ut_cons(cons, sigma) * w * deriv * e - \
        marg_ut_labor(n, lab_params)

    return euler

def solve_c(guess, params):
    '''
    Computes Euler errors for the FOC for savings in the steady state.
    This function is usually looped through over J, so it does one
    lifetime income group at a time.
    '''
    cons2, n2, b_splus1, r, w, T_H, BQ, theta, factor, e, sigma, beta, g_y, chi_b, tau_bq, rho, lambdas, J, S, \
        analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, method, s = params

    cons1 = guess

    # In order to not have 2 savings euler equations (one that solves the first S-1 equations, and one that solves the last one),
    # we combine them.  In order to do this, we have to compute a consumption term in period t+1, which requires us to have a shifted
    # e and n matrix.  We append a zero on the end of both of these so they will be the right size.  We could append any value to them,
    # since in the euler equation, the coefficient on the marginal utility of
    # consumption for this term will be zero (since rho is one).
    if method == 'TPI_scalar':
        # e_extended = np.array([e] + [0])
        # n_extended = np.array([n] + [0])
        etr_params_to_use = etr_params[S-s-1,:]
        mtry_params_to_use = mtry_params[S-s-1,:]
    else:
        # e_extended = np.array(list(e) + [0])
        # n_extended = np.array(list(n) + [0])
        # etr_params_to_use = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
        # mtry_params_to_use = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]
        etr_params_to_use = etr_params[S-s-1,:]
        mtry_params_to_use = mtry_params[S-s-1,:]

    # mtr_cap_params = (e_extended[1:], etr_params_to_use,
    #                   mtry_params_to_use,analytical_mtrs)
    mtr_cap_params = (e[S-s-1], etr_params_to_use,
                      mtry_params_to_use, analytical_mtrs)
    deriv = ((1+r) - r*(tax.MTR_capital(r, w, b_splus1, n_extended[1:], factor, mtr_cap_params)) -
             tax.tau_w_prime(b_splus1, (h_wealth, p_wealth, m_wealth))*b_splus1 -
             tax.tau_wealth(b_splus1, (h_wealth, p_wealth, m_wealth)))

    savings_ut = rho[S-s-2] * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)

    # Again, note timing in this equation, the (1-rho) term will zero out in the last period, so the last entry of cons2 can be complete
    # gibberish (which it is).  It just has to exist so cons2 is the right
    # size to match all other arrays in the equation.
    # print 'm_ut_cons1 = ', marg_ut_cons(cons1, sigma)
    # print 'm_ut_cons2 = ', marg_ut_cons(cons2, sigma)
    # print 'savings_ut = ', savings_ut
    # print 'c1 c2 bsp1= ', cons1, cons2, b_splus1

    euler_error = marg_ut_cons(cons1, sigma) - beta * (1 - rho[S-s-2]) * deriv * marg_ut_cons(
        cons2, sigma) * np.exp(-sigma * g_y) - savings_ut

    if cons1 <= 0:
        euler_error = 1e14


    return euler_error


def get_u(c, n, b_splus1, params):
    '''
    Computes flow utility for the household.

    Inputs:
        b_splus1 = [S,J] array, steady state distribution of capital
        n = [S,J] array, steady state distribution of labor
        c = [S,J] array, steady state distribution of consumption
        sigma = scalar, coefficient of relative risk aversion
        chi_n  = [S,] vector of utility weights for disulity of labor
        b_ellipse = scalar, scale parameter on elliptical utility
        ltilde = scalar, upper bound of household labor supply
        upsilon = scalar, curvature parameter on elliptical utility
        k_ellipse = scalar, shift parameter on elliptical utility
        rho_s = [S,] vector, mortality rates by age
        chi_b = [J,] vector, utility weights on bequests
        g_y = scalar, economic growth rate
        cons1 = guess

        Functions called: None

    Objects in function:
        utility = [S,J] array, utility for all agents

    Returns:
        utility
    '''
    sigma, chi_n, b_ellipse, ltilde, upsilon, rho_s, chi_b = params

    utility = (((c ** (1-sigma) - 1) / (1 - sigma)) +
               (chi_n * ((b_ellipse * (1 - (n / ltilde) ** upsilon)
                          ** (1 / upsilon)))) +
               (rho_s * chi_b * ((b_splus1 ** (1-sigma) - 1)
                                 / (1 - sigma))))

    return utility


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply in the steady state.

    Inputs:
        bssmat = [S,J] array, steady state distribution of capital
        nssmat = [S,J] array, steady state distribution of labor
        cssmat = [S,J] array, steady state distribution of consumption
        ltilde = scalar, upper bound of household labor supply

    Functions called: None

    Objects in function:
        flag2 = boolean, indicates if labor supply constraints violated (=False if not)

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on capital, labor, and consumption.'

    if (bssmat < 0).any():
        print '\tWARNING: There is negative capital stock'
    flag2 = False
    if (nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
        flag2 = True
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply along the transition path.
    Does this for each period t separately.

    Inputs:
        b_dist = [S,J] array, distribution of capital
        n_dist = [S,J] array, distribution of labor
        c_dist = [S,J] array, distribution of consumption
        t      = integer, time period
        ltilde = scalar, upper bound of household labor supply

    Functions called: None

    Objects in function: None

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    if (b_dist <= 0).any():
        print '\tWARNING: Aggregate capital is less than or equal to ' \
            'zero in period %.f.' % t
    if (n_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints ' \
            'in period %.f.' % t
    if (n_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint in '\
            'period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints in ' \
            'period %.f.' % t
