import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import factor, predictive, mutual_fund, common_dates, weighted_portfolio, FUND_RETURN
from regression import FactorModels
from graphs import tstat_graph, tstat_graph_by_category, pvalue_histogram
from computations import FDR


################################################################## REGRESSIONS ##################################################################

years_list = mutual_fund['year'].unique()
years_list = np.sort(years_list)
funds_list = mutual_fund['fundname'].unique()

df_alphas_uncondi = pd.DataFrame(index=funds_list, columns=years_list)
df_p_values_uncondi = pd.DataFrame(index=funds_list, columns=years_list)
# df_alphas_condi = pd.DataFrame(index=funds_list, columns=years_list)
# df_p_values_condi = pd.DataFrame(index=funds_list, columns=years_list)

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates_fund = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates_fund = pd.Index(sorted(list(common_dates_fund)))
    fund_years_list = common_dates_fund.str[:4].unique()
    
    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates_fund, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates_fund, FUND_RETURN] - factor.loc[common_dates_fund, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates_fund])
    four_factor = factor_models.four_factor_model()
    # conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    df_alphas_uncondi.loc[name, fund_years_list] = four_factor.params['const'] / len(fund_years_list)
    df_p_values_uncondi.loc[name, fund_years_list] = four_factor.pvalues['const'] # / len(fund_years_list)
    # df_alphas_condi.loc[name, fund_years_list] = conditional_four_factor.params['const'] / len(fund_years_list)
    # df_p_values_condi.loc[name, fund_years_list] = conditional_four_factor.pvalues['const'] # / len(fund_years_list)
    


######################################################## TEST FDR ANNUALISE ########################################################


# average_alphas_per_fund = df_alphas_uncondi.mean(axis=1) # Unconditionnal 4 factors model
# average_p_values_per_fund = df_p_values_uncondi.mean(axis=1) # Unconditionnal 4 factors model
# # average_alphas_per_fund = df_alphas_condi.mean(axis=1) # Conditionnal 4 factors model
# # average_p_values_per_fund = df_p_values_condi.mean(axis=1) # Conditionnal 4 factors model

# FDR_class = FDR(p_values=average_p_values_per_fund, alphas=average_alphas_per_fund, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
# fdr_annualized = FDR_class.compute_fdr()
# proportion_annualized = FDR_class.compute_proportions(nb_simul=1000)

# print("Results for compute FDR (uncondi) : ", fdr_annualized)
# print("Results for compute proportions (uncondi): ", proportion_annualized)

# Pareil que ce qu'on avait avant (normal)


############################################################ GRAPHIQUES ############################################################

### Graph 1 : 

df_alphas, df_p_values = df_alphas_uncondi.copy(), df_p_values_uncondi.copy() # Unconditionnal 4 factors model
# df_alphas, df_p_values = df_alphas_condi.copy(), df_p_values_condi.copy() # Conditionnal 4 factors model


nb_funds_per_year = df_alphas.count(axis=0)
dict_observed_proportions = {
    "zero_alpha funds": ((df_alphas > -1) & (df_alphas < 1)).sum(axis=0) / nb_funds_per_year, 
    "unskilled funds": (df_alphas < -1).sum(axis=0) / nb_funds_per_year, 
    "skilled funds": (df_alphas > 1).sum(axis=0) / nb_funds_per_year
}
df_observed_proportions = pd.DataFrame(dict_observed_proportions, index = years_list)
# print(df_observed_proportions)

proportions_fund_per_year = np.full(shape=(len(years_list), 3), fill_value=any)
year_i = 0
for year in years_list :
    FDR_year = FDR(p_values=df_p_values[year], alphas=df_alphas[year], gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
    proportions = FDR_year.compute_proportions(nb_simul=100)
    proportions_fund_per_year[year_i, 0] = proportions[0]
    proportions_fund_per_year[year_i, 1] = proportions[1]
    proportions_fund_per_year[year_i, 2] = proportions[2]
    year_i += 1
    
df_calculated_proportions = pd.DataFrame(proportions_fund_per_year, index=years_list, columns=['zero_alpha funds', 'unskilled funds', 'skilled funds'])
# print(df_calculated_proportions)


def graph_proportions(proportions:pd.DataFrame):
    plt.figure(figsize=(10, 5)) 
    plt.plot(proportions.index, proportions['zero_alpha funds'], marker='o', label='Zero Alpha Funds')
    plt.plot(proportions.index, proportions['skilled funds'], marker='o', label='Skilled Funds')
    plt.plot(proportions.index, proportions['unskilled funds'], marker='o', label='Unskilled Funds')

    plt.title('Number of Funds by Alpha Type Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Funds')
    plt.legend()
    plt.show()
    
graph_proportions(df_observed_proportions)
graph_proportions(df_calculated_proportions)

# ### Graph 2 : 

average_alphas_per_year = df_alphas.mean(axis=0)

def graph_alphas(timeline: np.ndarray, alphas: np.ndarray, nb_funds: np.ndarray):
    # Create a figure and a set of subplots.
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plotting the alphas with the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Alphas', color=color)
    ax1.plot(timeline, alphas, marker='o', color=color, label='Alphas')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for the number of funds
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Funds', color=color)
    ax2.plot(timeline, nb_funds, marker='o', color=color, label='Number of Funds')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Adding a title and a legend
    plt.title('Alphas and Number of Funds Over Time')
    fig.tight_layout() 
    fig.legend(loc='upper right')
    plt.show()

graph_alphas(years_list, average_alphas_per_year, nb_funds_per_year)

