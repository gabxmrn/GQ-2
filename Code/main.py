import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import factor, predictive, mutual_fund, weighted_portfolio
from regression import FactorModels
from graphs import tstat_graph, tstat_graph_by_category, pvalue_histogram
from computations import FDR


################################################################## REGRESSIONS ##################################################################

nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
fund_index = 0 # Counter for funds

results = np.full((nb_funds, 4), fill_value=np.nan) 

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates = pd.Index(sorted(list(common_dates)))
    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates, 'return'] - factor.loc[common_dates, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates])
    four_factor = factor_models.four_factor_model()
    conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 0] = four_factor.params['const']
    results[fund_index, 1] = four_factor.pvalues['const']
    results[fund_index, 2] = conditional_four_factor.params['const']
    results[fund_index, 3] = conditional_four_factor.pvalues['const']
    fund_index += 1

full_results = pd.DataFrame(results, index=fund_names, columns = ['alpha normal', 't-stat normal', 'alpha condi', 't-stat condi'])
# Create categories based on 'alpha'
full_results['Category normal'] = np.where(full_results['alpha normal'] > 1, 'pos', np.where(full_results['alpha normal'] < -1, 'neg', 'zero'))
full_results['Category condi'] = np.where(full_results['alpha condi'] > 1, 'pos', np.where(full_results['alpha condi'] < -1, 'neg', 'zero'))

#################################################################### GRAPHS #####################################################################

# Four factor model : 
tstat_graph(full_results, "t-stat normal") 
tstat_graph_by_category(full_results, "t-stat normal", "Category normal")
pvalue_histogram(full_results['Category normal'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)

# Conditional four factor model : 
tstat_graph(full_results, "t-stat condi") 
tstat_graph_by_category(full_results, "t-stat condi", "Category condi")
pvalue_histogram(full_results['Category condi'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)


###################################################################### FDR ######################################################################

pval_normal, alphas_normal = results[:, 1], results[:, 0]
pval_condi, alphas_condi = results[:, 3], results[:, 2]   

test_normal = FDR(p_values=pval_normal, alphas=alphas_normal, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
fdr_normal = test_normal.compute_fdr()
proportion_normal = test_normal.compute_proportions(nb_simul=1000)

test_condi = FDR(p_values=pval_condi, alphas=alphas_condi, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
fdr_condi = test_condi.compute_fdr()
proportion_condi = test_condi.compute_proportions(nb_simul=1000)

print("Results for compute FDR (normal) : ", fdr_normal)
print("Results for compute FDR (condi) : ", fdr_condi)
print("Results for compute proportions (normal): ", proportion_normal)
print("Results for compute proportions (condi): ", proportion_condi)




######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

# Portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)

# Caro -> A refaire pour avoir un joli tableau