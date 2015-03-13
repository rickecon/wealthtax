import pickle
import numpy

variables = pickle.load(open("given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

print bin_weights
print (Kssmat2*omega_SS).sum(0)/bin_weights

print factor_ss
