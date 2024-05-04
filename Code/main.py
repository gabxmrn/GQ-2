import pandas as pd
import numpy as np

from data import factor, predictive, mutual_fund, weighted_portfolio
from regression import FactorModels


######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

# Portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
# print(Portfolio_results.four_factor_model())
# print(Portfolio_results.conditional_four_factor_model())

# Remarque : la je ressort tous les coeff + les p-values qsui vont avec 
# LE CODE EST PLUS D'ACTUALITE AVEC LES CHANGEMENTS QUE J'AI FAIT AUJOURD'HUI

