import numpy as np 
import pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import scipy.optimize as opt 


# import pickle of baseline SS
variables = pickle.load(open('OUTPUT/SSinit/ss_init.pkl'))
for key in variables:
    globals()[key] = variables[key]


# Calculate expected lifetime utility by ability group


# import pickle of experiment 


def solver(LSRA)
    
#     Run SS with experiment and LSRA 

#     Calculate differences in utility from baseline 

#     return difference 

    pass

LSRA = np.ones(J)
LSRA = opt.fsolve(solver, LSRA)

plt.plot(np.arange(J), LSRA)
plt.savefig('LSRA')


