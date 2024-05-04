import pandas as pd
import numpy as np

from data import factor, predictive, mutual_fund, weighted_portfolio
from regression import FactorModels


######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

Portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
print(Portfolio_results.four_factor_model())
print(Portfolio_results.conditional_four_factor_model())

# Remarque : la je ressort tous les coeff + les p-values qsui vont avec 


########################################################### REGRESSIONS SUR LES FONDS ###########################################################

# nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
# fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
# fund_index = 0 # Counter for funds

# results = np.full((nb_funds, 5), fill_value=np.nan) 
# # Col1 = alpha model 1, Col2 = p-value model 1, Col3 = alpha model 2, Col4 = p-value model 2, Col5 = nb dates for fund

# for name, fund in mutual_fund.groupby('fundname'):
#     # Dates management : 
#     common_dates = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
#     common_dates = pd.Index(sorted(list(common_dates)))

#     # OLS : 
#     factor_models = FactorModels(exog = factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
#                                  endog = fund.loc[common_dates, 'rdt'] - factor.loc[common_dates, 'rf_rate'] , 
#                                  predictive = predictive.loc[common_dates])
#     results_model_1 = factor_models.four_factor_model()
#     results_model_2 = factor_models.conditional_four_factor_model()
    
#     # Résults : 
#     fund_names[fund_index] = name
#     results[fund_index, 0] = results_model_1.loc['const', 'Coeff']
#     results[fund_index, 1] = results_model_1.loc['const', 'P-value']
#     results[fund_index, 2] = results_model_2.loc['const', 'Coeff']
#     results[fund_index, 3] = results_model_2.loc['const', 'P-value']
#     results[fund_index, 4] = len(common_dates)
#     fund_index += 1

# full_results = pd.DataFrame(results, index=fund_names, columns = ["alpha 1", "p-value 1", "alpha 2", "p-value 2","number of data"])
# print(full_results.head(20))

# Remarque : la je me concentre que sur alpha + pvalue (pour les deux modèles)