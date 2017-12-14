========================
Wealth Tax Paper
========================

Abstract
========
This project looks at the effects of both a wealth tax and an increase in the progressivity of the income tax on inequality in the context of an OLG model that is closely related to the open source [OG-USA model](https://github.com/open-source-economics/OG-USA).  The repo includes data used in the calibration, and all the Python code necessary to solve and simulate the model.

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
In order to run the model, please do the following (after making sure you have the Anaconda distribution of Python installed):
1. Set your Pyton environment to run the scripts in this repos:
    * `conda env create`
    * `source activate ospcdyn`
2. Navigate to the `Python` folder and run
    * `python run_wealthtax.py`
3. To produce output after these runs have completed exectute:
    * `python wealthtax_figures.py`
    * `python wealthtax_tables.py`
    * The resulting output will be in `WealthTaxTables.xlsx` and in the `Graphs` folder.
