import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import factor, predictive, mutual_fund, common_dates, weighted_portfolio, FUND_RETURN
from regression import FactorModels
from graphs import tstat_graph, tstat_graph_by_category, pvalue_histogram
from computations import FDR


################################################################## REGRESSIONS ##################################################################

nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
nb_dates = len(common_dates) # Total number of dates
print(f"Number of funds : {nb_funds}, Number of dates : {nb_dates}")

fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
fund_index = 0 # Counter for funds

results = np.full((nb_funds, 4), fill_value=np.nan) 

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates_fund = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates_fund = pd.Index(sorted(list(common_dates_fund)))
    
    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates_fund, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates_fund, FUND_RETURN] - factor.loc[common_dates_fund, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates_fund])
    four_factor = factor_models.four_factor_model()
    # conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 0] = four_factor.params['const']
    results[fund_index, 1] = four_factor.pvalues['const']
    # results[fund_index, 2] = conditional_four_factor.params['const']
    # results[fund_index, 3] = conditional_four_factor.pvalues['const']
    fund_index += 1

full_results = pd.DataFrame(results, index=fund_names, columns = ['alpha uncondi', 't-stat uncondi', 'alpha condi', 't-stat condi'])
# Create categories based on 'alpha'
full_results['Category uncondi'] = np.where(full_results['alpha uncondi'] > 1, 'pos', np.where(full_results['alpha uncondi'] < -1, 'neg', 'zero'))
# full_results['Category condi'] = np.where(full_results['alpha condi'] > 1, 'pos', np.where(full_results['alpha condi'] < -1, 'neg', 'zero'))


#################################################################### GRAPHS #####################################################################

# Four factor model : 
tstat_graph(full_results, "t-stat uncondi") 
tstat_graph_by_category(full_results, "t-stat uncondi", "Category uncondi")
pvalue_histogram(full_results['Category uncondi'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)

# # Conditional four factor model : 
# tstat_graph(full_results, "t-stat condi") 
# tstat_graph_by_category(full_results, "t-stat condi", "Category condi")
# pvalue_histogram(full_results['Category condi'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)


###################################################################### FDR ######################################################################

pval_uncondi, alphas_uncondi = results[:, 1], results[:, 0]
# pval_condi, alphas_condi = results[:, 3], results[:, 2]   

test_uncondi = FDR(p_values=pval_uncondi, alphas=alphas_uncondi, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
fdr_uncondi = test_uncondi.compute_fdr()
proportion_uncondi = test_uncondi.compute_proportions(nb_simul=1000)

# test_condi = FDR(p_values=pval_condi, alphas=alphas_condi, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
# fdr_condi = test_condi.compute_fdr()
# proportion_condi = test_condi.compute_proportions(nb_simul=1000)

print("Results for compute FDR (uncondi) : ", fdr_uncondi)
# print("Results for compute FDR (condi) : ", fdr_condi)
print("Results for compute proportions (uncondi): ", proportion_uncondi)
# print("Results for compute proportions (condi): ", proportion_condi)


#################################################################### TABLEAU ####################################################################





######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

# portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
# print(portfolio_results)

# Caro -> A refaire pour avoir un joli tableau