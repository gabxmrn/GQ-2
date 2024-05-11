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

results = np.full((nb_funds, 6), fill_value=np.nan) 

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates_fund = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates_fund = pd.Index(sorted(list(common_dates_fund)))
    
    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates_fund, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates_fund, FUND_RETURN] - factor.loc[common_dates_fund, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates_fund])
    four_factor = factor_models.four_factor_model()
    conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 0] = four_factor.params['const']
    results[fund_index, 1] = four_factor.pvalues['const']
    results[fund_index, 2] = four_factor.tvalues['const']
    results[fund_index, 3] = conditional_four_factor.params['const']
    results[fund_index, 4] = conditional_four_factor.pvalues['const'] 
    results[fund_index, 5] = conditional_four_factor.tvalues['const'] 
    fund_index += 1

full_results = pd.DataFrame(results, index=fund_names, columns = ['alpha uncondi', 'pvalues uncondi','t-stat uncondi', 'alpha condi', 'pvalues condi', 't-stat condi'])
# Create categories based on 'alpha'
full_results['Category uncondi'] = np.where(full_results['alpha uncondi'] > 1, 'pos', np.where(full_results['alpha uncondi'] < -1, 'neg', 'zero'))
full_results['Category condi'] = np.where(full_results['alpha condi'] > 1, 'pos', np.where(full_results['alpha condi'] < -1, 'neg', 'zero'))


#################################################################### GRAPHS #####################################################################

# Four factor model : 
tstat_graph(full_results, "pvalues uncondi") 
tstat_graph_by_category(full_results, "pvalues uncondi", "Category uncondi")
pvalue_histogram(full_results['Category uncondi'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)

# Conditional four factor model : 
tstat_graph(full_results, "pvalues condi") 
tstat_graph_by_category(full_results, "pvalues condi", "Category condi")
pvalue_histogram(full_results['Category condi'].value_counts() / nb_funds, [0, -2.5, 3], 1, nb_funds)


###################################################################### FDR ######################################################################

pval_uncondi, alphas_uncondi, t_stat_uncondi = results[:, 1], results[:, 0], results[:, 2]
pval_condi, alphas_condi, t_stat_condi = results[:, 4], results[:, 3], results[:, 5] 

test_uncondi = FDR(p_values=pval_uncondi, alphas=alphas_uncondi, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
fdr_uncondi = test_uncondi.compute_fdr()
proportion_uncondi = test_uncondi.compute_proportions(nb_simul=1000)

test_condi = FDR(p_values=pval_condi, alphas=alphas_condi, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
fdr_condi = test_condi.compute_fdr()
proportion_condi = test_condi.compute_proportions(nb_simul=1000)

# print("Results for compute FDR (uncondi) : ", fdr_uncondi)
# print("Results for compute FDR (condi) : ", fdr_condi)
# print("Results for compute proportions (uncondi): ", proportion_uncondi)
# print("Results for compute proportions (condi): ", proportion_condi)
bias_test_u = test_uncondi.compute_bias(t_stats=t_stat_uncondi, T=nb_dates)
print("Results for compute bias (uncondi) : ", bias_test_u)
bias_test_c = test_uncondi.compute_bias(t_stats=t_stat_condi, T=nb_dates)
print("Results for compute bias (condi) : ", bias_test_c)


bias_test_s = test_uncondi.compute_bias_simple(expected_pi0= 0.75)
print("Results for compute bias simple (uncondi) : ", bias_test_s)

#################################################################### TABLEAU ####################################################################





######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

# portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
# print(portfolio_results)

# Caro -> A refaire pour avoir un joli tableau