'''
------------------------------------------------------------------------
Last updated: 2/13/2015

Calculates steady state of OLG model with S age cohorts, using simple
bin_weights and no taxes in order to be able to converge in the steady
state, in order to get initial values for later simulations

This py-file calls the following other file(s):
            income.py
            demographics.py
            tax_funcs.py
            OUTPUT/given_params.pkl
            OUTPUT/Nothing/wealth_data_moments.pkl
            OUTPUT/Nothing/{}.pkl
                name depends on what iteration just ran

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Nothing/{}.pkl
                name depends on what iteration is being run
            OUTPUT/Nothing/init_init_sols.pkl
            OUTPUT/Nothing/payroll_inputs.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import time
import os
import scipy.optimize as opt
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import income
from scipy import stats
import demographics
import tax_funcs as tax


'''
------------------------------------------------------------------------
Imported user given values
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
bin_weights_init  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
A            = total factor productivity parameter in firms' production
               function
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor that changes with S (Sx1 array)
chi_b        = discount factor of incidental bequests
eta          = Frisch elasticity of labor supply
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
omega_ellipse= value of omega for elliptical fit of utility function
slow_work    = time at which chi_n starts increasing from 1
mean_income  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
tau_sales    = sales tax (scalar)
tau_bq       = bequest tax (scalar)
tau_lump     = lump sum tax (scalar)
tau_payroll  = payroll tax (scalar)
theta_tax    = payback value for payroll tax (scalar)
retire       = age in which individuals retire(scalar)
h_wealth     = wealth tax parameter h
p_wealth     = wealth tax parameter p
m_wealth     = wealth tax parameter m
thetas_simulation = is this the last simulation before getting
                replacement rates
scal         = value to scale the initial guesses by in order to get the
               fsolve to converge
name_of_last = name of last simulation that ran
name_of_it   = name of current simulation
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/Nothing/wealth_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("OUTPUT/Nothing/labor_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
bin_weights = bin_weights_init
'''
------------------------------------------------------------------------
Generate income and demographic parameters
------------------------------------------------------------------------
e            = S x J matrix of age dependent possible working abilities
               e_s
omega        = T x S x J array of demographics
g_n          = steady state population growth rate
omega_SS     = steady state population distribution
children     = T x starting_age x J array of children demographics
surv_rate    = S x 1 array of survival rates
mort_rate    = S x 1 array of mortality rates
------------------------------------------------------------------------
'''

omega, g_n, omega_SS, children, surv_rate = demographics.get_omega(
    S, J, T, bin_weights, starting_age, ending_age, E)
e = income.get_e(S, J, starting_age, ending_age, bin_weights, omega_SS)
mort_rate = 1-surv_rate


# chi_n_guess = np.ones(S) * 1.0
# slow_work = np.round(7.0 * S / 16.0)
# chi_n_multiplier = 30
# chi_n_guess[slow_work:] = (mort_rate[slow_work:] + 1 - mort_rate[slow_work])**chi_n_multiplier
# chi_n_guess[-14:] = chi_n_guess[-14]

# chi_n_guess = np.array([25.0  , 23.36808929  , 19.0978715   , 11.91693867
#   ,  9.39244751  ,  8.04772528  ,  7.13328096  ,  6.6569549   ,  6.18471577
#   ,  5.79413131  ,  5.57810188  ,  5.28029037  ,  5.09000427  ,  4.92017985
#   ,  4.68293528  ,  4.50917643  ,  4.34630557  ,  4.19698782  ,  4.07292637
#   ,  4.00262846  ,  3.8769443   ,  3.79018663  ,  3.71971047  ,  3.62175787
#   ,  3.57837772  ,  3.51694841  ,  3.40408528  ,  3.40313675  ,  3.35271903
#   ,  3.34590638  ,  3.35717063  ,  3.37030627  ,  3.41669435  ,  3.44098278
#   ,  3.50808345  ,  3.5794458   ,  3.7274887   ,  3.77750174  ,  3.90881753
#   ,  4.08481377  ,  4.32855403  ,  4.99120077  ,  5.66381587  ,  6.12273368
#   ,  7.03008813  ,  8.05988511  ,  8.58717935  ,  8.94846604  ,  9.60453523
#   ,  9.88782496  ,  9.94918009  , 10.32850955  , 10.30528814  ,  9.96002746
#   ,  9.95174946  , 10.10262544  ,  9.90165704  ,  9.78845576  ,  9.81118447
#   ,  9.85190971  ,  9.92320211  , 10.01857802  , 10.10315541  , 10.21539475
#   , 10.38340607  , 10.57052623  , 10.7608433   , 11.0624123   , 11.34197591
#   , 11.65211931  , 11.99754856  , 12.39999565  , 12.89778177  , 13.3494677
#   , 13.8837629   , 14.45447548  , 15.10650203  , 14.55039195  , 15.1345734
#   , 15.47871223])

chi_n_guess = np.array([46.43103569 ,  25.40267722 ,  14.28366036 ,  10.83368034
 ,   8.45625515 ,   7.11317213 ,   6.52761155 ,   6.01693931 ,   5.46449885
 ,   5.09782791 ,   4.87555018 ,   4.60012673 ,   4.45359059 ,   4.31047208
 ,   4.05662526 ,   4.04725308 ,   3.72016851 ,   2.7593443  ,   3.61187287
 ,   3.48775222 ,   3.37069344 ,   3.30025423 ,   3.30317224 ,   3.03571148
 ,   3.27364837 ,   3.22723277 ,   3.04089574 ,   3.16802732 ,   3.02408407
 ,   2.84869197 ,   3.07992421 ,   3.09463504 ,   3.13974939 ,   3.20918915
 ,   3.29703147 ,   3.28266044 ,   3.67399162 ,   3.66482653 ,   3.78750891
 ,   3.99646818 ,   4.31163041 ,   5.00344695 ,   6.11371906 ,   6.29349741
 ,   7.27272655 ,   8.48719262 ,   9.37100285 ,   9.83132055 ,  10.49184416
 ,  10.97337187 ,  11.12050979 ,  11.77673893 ,  11.7878914  ,  11.71717301
 ,  11.58458214 ,  11.76908197 ,  11.45466426 ,  11.71191348 ,  11.89606755
 ,  12.33591946 ,  12.11880122 ,  12.67935938 ,  12.3145382  ,  12.36754116
 ,  13.4839303  ,  12.40639494 ,  12.54946963 ,  13.53984928 ,  12.66873465
 ,  12.99913171 ,  13.24694181 ,  13.37659492 ,  13.76147868 ,  13.70442569
 ,  14.38740359 ,  14.37493147 ,  14.51253107 ,  14.53498107 ,  15.11107412
 ,  15.2181688])


surv_rate[-1] = 0.0
mort_rate[-1] = 1


'''
------------------------------------------------------------------------
Finding the Steady State
------------------------------------------------------------------------
K_guess_init = (S-1 x J) array for the initial guess of the distribution
               of capital
L_guess_init = (S x J) array for the initial guess of the distribution
               of labor
solutions    = ((S * (S-1) * J * J) x 1) array of solutions of the
               steady state distributions of capital and labor
Kssmat       = ((S-1) x J) array of the steady state distribution of
               capital
Kssmat2      = SxJ array of capital (zeros appended at the end of
               Kssmat2)
Kssmat3      = SxJ array of capital (zeros appended at the beginning of
               Kssmat)
Kssvec       = ((S-1) x 1) vector of the steady state level of capital
               (averaged across ability types)
Kss          = steady state aggregate capital stock
K_agg        = Aggregate level of capital
Lssmat       = (S x J) array of the steady state distribution of labor
Lssvec       = (S x 1) vector of the steady state level of labor
               (averaged across ability types)
Lss          = steady state aggregate labor
Yss          = steady state aggregate output
wss          = steady state real wage
rss          = steady state real rental rate
cssmat       = SxJ array of consumption across age and ability groups
runtime      = Time needed to find the steady state (seconds)
hours        = Hours needed to find the steady state
minutes      = Minutes needed to find the steady state, less the number
               of hours
seconds      = Seconds needed to find the steady state, less the number
               of hours and minutes
------------------------------------------------------------------------
'''

# Functions and Definitions


def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = A * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    return Y_now


def get_w(Y_now, L_now):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w_now = (1 - alpha) * Y_now / L_now
    return w_now


def get_r(Y_now, K_now):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_L(e, n):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    L_now = np.sum(e * omega_SS * n)
    return L_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(n, chi_n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n.reshape(S, 1) * deriv
    return output


def MUb(chi_b, bq):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b[-1] * (bq ** (-sigma))
    return output


def get_cons(r, k1, w, e, lab, bq, bins, k2, gy, tax):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, bin_weights, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    cons = (1 + r)*k1 + w*e*lab + bq / bins - k2*np.exp(gy) - tax
    return cons


def Euler1(w, r, e, L_guess, K1, K2, K3, B, factor, taulump, chi_b):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K1       = distribution of capital in period t ((S-1) x J array)
        K2       = distribution of capital in period t+1 ((S-1) x J array)
        K3       = distribution of capital in period t+2 ((S-1) x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        taulump  = lump sum transfer from the government to the households
        xi       = coefficient of relative risk aversion
        chi_b    = discount factor of savings

    Returns:
        Value of Euler error.
    '''
    B_euler = B.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, K1, w, e[:-1, :], L_guess[:-1, :], B_euler, bin_weights, factor, taulump)
    tax2 = tax.total_taxes_SS2(r, K2, w, e[1:, :], L_guess[1:, :], B_euler, bin_weights, factor, taulump)
    cons1 = get_cons(r, K1, w, e[:-1, :], L_guess[:-1, :], B_euler, bin_weights, K2, g_y, tax1)
    cons2 = get_cons(r, K2, w, e[1:, :], L_guess[1:, :], B_euler, bin_weights, K3, g_y, tax2)
    income = (r * K2 + w * e[1:, :] * L_guess[1:, :]) * factor
    deriv = (
        1 + r*(1-tax.tau_income(r, K1, w, e[1:, :], L_guess[1:, :], factor)-tax.tau_income_deriv(
            r, K1, w, e[1:, :], L_guess[1:, :], factor)*income)-tax.tau_w_prime(K2)*K2-tax.tau_wealth(K2))
    bequest_ut = (1-surv_rate[:-1].reshape(S-1, 1)) * np.exp(-sigma * g_y) * chi_b[:-1].reshape(S-1, 1) * K2 ** (-sigma)
    euler = MUc(cons1) - beta * surv_rate[:-1].reshape(S-1, 1) * deriv * MUc(
        cons2) * np.exp(-sigma * g_y) - bequest_ut
    return euler


def Euler2(w, r, e, L_guess, K1_2, K2_2, B, factor, taulump, chi_n):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K1_2     = distribution of capital in period t (S x J array)
        K2_2     = distribution of capital in period t+1 (S x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        taulump  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    B = B.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, K1_2, w, e, L_guess, B, bin_weights, factor, taulump)
    cons = get_cons(r, K1_2, w, e, L_guess, B, bin_weights, K2_2, g_y, tax1)
    income = (r * K1_2 + w * e * L_guess) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, K1_2, w, e, L_guess, factor) - tax.tau_income_deriv(
        r, K1_2, w, e, L_guess, factor) * income
    euler = MUc(cons) * w * deriv * e - MUl(L_guess, chi_n)
    return euler


def Euler3(w, r, e, L_guess, K_guess, B, factor, chi_b, taulump):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K_guess  = distribution of capital in period t (S-1 x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        chi_b    = discount factor of savings
        taulump  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    B = B.reshape(1, J)
    tax1 = tax.total_taxes_eul3_SS(r, K_guess[-2, :], w, e[-1, :], L_guess[-1, :], B, bin_weights, factor, taulump)
    cons = get_cons(r, K_guess[-2, :], w, e[-1, :], L_guess[-1, :], B, bin_weights, K_guess[-1, :], g_y, tax1)
    euler = MUc(cons) - np.exp(-sigma * g_y) * MUb(
        chi_b, K_guess[-1, :])
    return euler


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data and
        simulated data
    '''
    frac = (simul - data)/data
    # frac *= 100
    # output = frac ** 2
    output = np.abs(frac)
    return output


def Steady_State(guesses, params):
    '''
    Parameters: Steady state distribution of capital guess as array
                size 2*S*J

    Returns:    Array of 2*S*J Euler equation errors
    '''
    chi_b = params[0]
    chi_b *= np.ones(S)
    chi_n = np.array(params[1:])
    K_guess = guesses[0: S * J].reshape((S, J))
    B = (K_guess * omega_SS * mort_rate.reshape(S, 1)).sum(0)
    K = (omega_SS * K_guess).sum()
    L_guess = guesses[S * J:-1].reshape((S, J))
    L = get_L(e, L_guess)
    Y = get_Y(K, L)
    w = get_w(Y, L)
    r = get_r(Y, K)
    BQ = (1 + r) * B
    K1 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess[:-2, :]))
    K2 = K_guess[:-1, :]
    K3 = K_guess[1:, :]
    K1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess[:-1, :]))
    K2_2 = K_guess
    factor = guesses[-1]
    taulump = tax.tax_lump(r, K1_2, w, e, L_guess, BQ, bin_weights, factor, omega_SS)
    error1 = Euler1(w, r, e, L_guess, K1, K2, K3, BQ, factor, taulump, chi_b)
    error2 = Euler2(w, r, e, L_guess, K1_2, K2_2, BQ, factor, taulump, chi_n)
    error3 = Euler3(w, r, e, L_guess, K_guess, BQ, factor, chi_b, taulump)
    avI = ((r * K1_2 + w * e * L_guess) * omega_SS).sum()
    error4 = [mean_income - factor * avI]
    # Check and punish constraint violations
    mask1 = L_guess < 0
    error2[mask1] += 1e9
    mask2 = L_guess > ltilde
    error2[mask2] += 1e9
    if K_guess.sum() <= 0:
        error1 += 1e9
    tax1 = tax.total_taxes_SS(r, K1_2, w, e, L_guess, BQ, bin_weights, factor, taulump)
    cons = get_cons(r, K1_2, w, e, L_guess, BQ.reshape(1, J), bin_weights, K2_2, g_y, tax1)
    mask3 = cons < 0
    error2[mask3] += 1e9
    # print np.abs(np.array(list(error1.flatten()) + list(
    #     error2.flatten()) + list(error3.flatten()) + error4)).max()
    return list(error1.flatten()) + list(
        error2.flatten()) + list(error3.flatten()) + error4


def borrowing_constraints(K_dist, w, r, e, n, BQ):
    '''
    Parameters:
        K_dist = Distribution of capital ((S-1)xJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        n      = distribution of labor (SxJ array)
        BQ     = bequests

    Returns:
        False value if all the borrowing constraints are met, True
            if there are violations.
    '''
    b_min = np.zeros((S-1, J))
    b_min[-1, :] = (ctilde + bqtilde - w * e[S-1, :] * ltilde - BQ.reshape(
        1, J) / bin_weights) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + np.exp(g_y) * b_min[-(i+1), :] - w * e[
            -(i+2), :] * ltilde - BQ.reshape(1, J) / bin_weights) / (1 + r)
    difference = K_dist - b_min
    if (difference < 0).any():
        return True
    else:
        return False


def constraint_checker(Kssmat, Lssmat, wss, rss, e, cssmat, BQ):
    '''
    Parameters:
        Kssmat = steady state distribution of capital ((S-1)xJ array)
        Lssmat = steady state distribution of labor (SxJ array)
        wss    = steady state wage rate (scalar)
        rss    = steady state rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        cssmat = steady state distribution of consumption (SxJ array)
        BQ     = bequests

    Created Variables:
        flag1 = False if all borrowing constraints are met, true
               otherwise.
        flag2 = False if all labor constraints are met, true otherwise

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    # print 'Checking constraints on capital, labor, and consumption.'
    flag1 = False
    if Kssmat.sum() <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if borrowing_constraints(Kssmat, wss, rss, e, Lssmat, BQ) is True:
        print '\tWARNING: Borrowing constraints have been violated.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (Lssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (Lssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption volates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def func_to_min(bq_guesses_init, other_guesses_init):
    '''
    Parameters:
        bq_guesses_init = guesses for chi_b
        other_guesses_init = guesses for the distribution of capital and labor
                            stock, and factor value

    Returns:
        The max absolute deviation between the actual and simulated
            wealth moments
    '''
    print bq_guesses_init
    Steady_State_X = lambda x: Steady_State(x, bq_guesses_init)
    solutions = opt.fsolve(Steady_State_X, other_guesses_init, xtol=1e-13)
    K_guess = solutions[0: S * J].reshape((S, J))
    K2 = K_guess[:-1, :]
    factor = solutions[-1]
    # Wealth Calibration Euler
    p99_sim = K2[:, -1] * factor
    bq_half = perc_dif_func(p99_sim[:-3], p99[2:])
    bq_tomatch = bq_half[11:45]
    error5 = list(bq_tomatch)
    # labor calibration euler
    labor_sim = ((solutions[S*J:2*S*J]).reshape(S, J)*bin_weights.reshape(1, J)).sum(axis=1)
    error6 = list(perc_dif_func(labor_sim, labor_dist_data))
    # combine eulers
    output = np.array(error5 + error6)
    fsolve_no_converg = np.abs(Steady_State_X(solutions)).max()
    if np.isnan(fsolve_no_converg):
        fsolve_no_converg = 1e6
    if fsolve_no_converg > 1e-4:
        output += 1e9
    if (bq_guesses_init <= 0.0).any():
        output += 1e9
    print output.sum()
    return output.sum()

starttime = time.time()
if name_of_last != 'none':
    variables = pickle.load(open("OUTPUT/Nothing/{}.pkl".format(name_of_last), "r"))
    for key in variables:
        globals()[key] = variables[key]
    guesses = list((solutions[:S*J].reshape(S, J) * scal).flatten()) + list(
        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
else:
    K_guess_init = np.ones((S, J)) * .01
    L_guess_init = np.ones((S, J)) * .99 * ltilde
    Kg = (omega_SS * K_guess_init).sum()
    Lg = get_L(e, L_guess_init)
    Yg = get_Y(Kg, Lg)
    wguess = get_w(Yg, Lg)
    rguess = get_r(Yg, Kg)
    avIguess = ((rguess * K_guess_init + wguess * e * L_guess_init) * omega_SS).sum()
    factor_guess = [mean_income / avIguess]
    guesses = list(K_guess_init.flatten()) + list(L_guess_init.flatten()) + factor_guess




if 'final_bq_params' in globals():
    bq_guesses = final_bq_params
else:
    # bq_guesses = np.ones(S+1) * 1.0
    bq_guesses = np.ones(S+1) * 88.78340244
    bq_guesses[1:] = chi_n_guess
    bq_guesses = list(bq_guesses)
func_to_min_X = lambda x: func_to_min(x, guesses)

if thetas_simulation:
    final_bq_params = opt.minimize(func_to_min_X, bq_guesses, method='SLSQP').x
    print 'The final bequest parameter values:', final_bq_params
else:
    final_bq_params = bq_guesses
Steady_State_X2 = lambda x: Steady_State(x, final_bq_params)
solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
print np.array(Steady_State_X2(solutions)).max()

# Save the solutions of SS
if thetas_simulation is False:
    var_names = ['solutions', 'final_bq_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/{}.pkl".format(name_of_it), "w"))
else:
    var_names = ['solutions', 'final_bq_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/init_init_sols.pkl", "w"))
    runtime = time.time() - starttime
    hours = runtime / 3600
    minutes = (runtime / 60) % 60
    seconds = runtime % 60
    Kssmat = solutions[0:(S-1) * J].reshape(S-1, J)
    BQ = solutions[(S-1)*J:S*J]
    Bss = (np.array(list(Kssmat) + list(BQ.reshape(1, J))).reshape(
        S, J) * omega_SS * mort_rate.reshape(S, 1)).sum(0)
    Kssmat2 = np.array(list(np.zeros(J).reshape(1, J)) + list(Kssmat))
    Kssmat3 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))

    Kssvec = Kssmat.sum(1)
    Kss = (omega_SS[:-1, :] * Kssmat).sum() + (omega_SS[-1, :]*BQ).sum()
    Kssavg = Kssvec.mean()
    Kssvec = np.array([0]+list(Kssvec))
    Lssmat = solutions[S * J:-1].reshape(S, J)
    Lssvec = Lssmat.sum(1)
    Lss = get_L(e, Lssmat)
    Lssavg = Lssvec.mean()
    Yss = get_Y(Kss, Lss)
    wss = get_w(Yss, Lss)
    rss = get_r(Yss, Kss)
    k1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat))
    factor_ss = solutions[-1]
    B = Bss * (1+rss)
    Tss = tax.tax_lump(rss, Kssmat2, wss, e, Lssmat, B, bin_weights, factor_ss, omega_SS)
    taxss = tax.total_taxes_SS(rss, Kssmat2, wss, e, Lssmat, B, bin_weights, factor_ss, Tss)
    cssmat = get_cons(rss, Kssmat2, wss, e, Lssmat, (1+rss)*B.reshape(1, J), bin_weights.reshape(1, J), Kssmat3, g_y, taxss)

    constraint_checker(Kssmat, Lssmat, wss, rss, e, cssmat, BQ)

    '''
    ------------------------------------------------------------------------
    Generate variables for graphs
    ------------------------------------------------------------------------
    k1          = (S-1)xJ array of Kssmat in period t-1
    k2          = copy of Kssmat
    k3          = (S-1)xJ array of Kssmat in period t+1
    k1_2        = SxJ array of Kssmat in period t
    k2_2        = SxJ array of Kssmat in period t+1
    euler1      = euler errors from first euler equation
    euler2      = euler errors from second euler equation
    euler3      = euler errors from third euler equation
    ------------------------------------------------------------------------
    '''
    k1 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat[:-1, :]))
    k2 = Kssmat
    k3 = np.array(list(Kssmat[1:, :]) + list(BQ.reshape(1, J)))
    k1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat))
    k2_2 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))

    K_eul3 = np.zeros((S, J))
    K_eul3[:S-1, :] = Kssmat
    K_eul3[-1, :] = BQ
    chi_b = final_bq_params[0]
    chi_b *= np.ones(S)
    chi_n = np.array(final_bq_params[1:])
    euler1 = Euler1(wss, rss, e, Lssmat, k1, k2, k3, B, factor_ss, Tss, chi_b)
    euler2 = Euler2(wss, rss, e, Lssmat, k1_2, k2_2, B, factor_ss, Tss, chi_n)
    euler3 = Euler3(wss, rss, e, Lssmat, K_eul3, B, factor_ss, chi_b, Tss)

    '''
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    '''

    Kssmat_init = np.array(list(Kssmat) + list(BQ.reshape(1, J)))
    Lssmat_init = Lssmat
    var_names = ['retire', 'Lssmat_init', 'wss', 'factor_ss', 'e',
                 'J', 'omega_SS']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/payroll_inputs.pkl", "w"))
