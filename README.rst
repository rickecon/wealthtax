========================
Piketty Wealth Tax Paper
========================

A Brigham Young University Macroeconomics and Computational Laboratory project in conjunction with the Open Source Policy Center at the American Enterprise Institute.

Abstract
========
This project looks at the effects of both a wealth tax and an increase in the progressivity of the income tax on inequality in the context of an OLG model that is closely related to the dyanamic scoring model being developed by the BYUMCL and AEI's OSPC.  The repo includes data used in the calibration, and all the Python code necessary to solve and simulate the model.

Contributors
============
- Jason DeBacker
- Richard W. Evans
- Evan Magnusson
- Kerk Phillips
- Shanthi Ramnath
- Isaac Swift

How to use the Repository
=========================
In order to simulate the model, first run the 'run_baseline.py' file in the wealthtax/Python folder.  This will simulate the baseline steady-state and time path.  Then, in the 'wealthtax/Python/OUTPUT' folder, run 'wealth_fit_graphs.py' to see the scaling factor that transforms the moments to dollar terms and the average wealth levels for each ability type.  Using these values, open 'Effective Wealth Tax Rates.xlsx' in the 'wealthtax/Data' folder. Replace the f and average wealth levels, and run the solver to solve for h and m.  Then, replace these values in 'Python/run_tax_experiments.py' for h_wealth and m_wealth.  Then, run 'run_tax_experiments.py' in order to simulate the steady-state and time paths of the model with the wealth tax and calibrated income tax.  
To generate the graphs, run 'SS_graphs.py' and 'TPI_graphs.py' in the 'wealthtax/Python/OUTPUT_income_tax' and 'wealthtax/Python/OUTPUT_wealth_tax' to generate steady-state and TPI graphs for the income tax and wealth tax experiments, respectively.  The baseline graphs will be in both folders.  'wealth_read.py' in these subfolders generates the graphs of the wealth fit, and 'alt_inequality_measures.py' outputs the alternate measures of inequality for the paper.  'gini_grapher.py' graphs GINI plots for the baseline and both tax experiments side by side.
