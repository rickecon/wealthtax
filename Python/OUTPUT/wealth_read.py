import pickle
import numpy as np

variables = pickle.load(open("SSinit/ss_init.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("Nothing/wealth_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

# print bin_weights
# print (Kssmat2*omega_SS).sum(0)/bin_weights

# print factor_ss

# print chi_b

for j in xrange(7):
    print 'j=', j+1
    print factor_ss*(Kssmat2*omega_SS)[1:21, j].sum()/omega_SS[1:21, j].sum()
    print factor_ss*(Kssmat2*omega_SS)[21:46, j].sum()/omega_SS[21:46, j].sum()
    print factor_ss*(Kssmat2*omega_SS)[46:, j].sum()/omega_SS[46:, j].sum()


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

domain = np.linspace(20, 95, 76)
plt.plot(domain, highest_wealth_data[2:])
plt.plot(domain, factor_ss * Kssmat[:76, -1])
plt.show()