import pandas as pd


################################################ DATA IMPORTATION ###############################################

exog_path = r"Data\Clean\exogeneous_variables.xlsx"
# mf_path = r"Data\Clean\tr_mutualfunds S12.csv"

# Exogeneous variables
exog = pd.read_excel(exog_path, index_col=('Dates'))

# Mutual funds
# mf = pd.read_csv(mf_path)

################################################ DATA PROCESSING ###############################################

# Predictive variables of the conditionnal 4 factors model : value at end of month t - mean on the period
for column in exog.columns:
    if column in ["1M_Tbill_yield", "div_yield_mkt_ptf", "term_spread", "default_spread"]:
        exog[column] -= exog[column].mean()
