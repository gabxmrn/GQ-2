import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import factor, predictive, mutual_fund, common_dates, weighted_portfolio
from regression import FactorModels
from graphs import tstat_graph, tstat_graph_by_category, pvalue_histogram
from computations import FDR


################################################################## REGRESSIONS ##################################################################

# Regression par fond et par an -> alphas annualisés 

years_list = mutual_fund['year'].unique()
funds_list = mutual_fund['fundname'].unique()

df_alphas = pd.DataFrame(index=funds_list, columns=years_list)
df_p_values = pd.DataFrame(index=funds_list, columns=years_list)

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates_fund = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates_fund = pd.Index(sorted(list(common_dates_fund)))
    years_list_fund = common_dates_fund.str[:4]
    
    # OLS : 
    for year in years_list_fund :
        year_indices = common_dates_fund[common_dates_fund.str.startswith(year)]

        if len(year_indices) < 3 :
            # print(f"Not enough data for fund {name} in year {year}")
            continue
        
        factor_models = FactorModels(
            exog=factor.loc[year_indices, ['mkt_rf', 'smb', 'hml', 'mom']],
            endog=fund.loc[year_indices, 'return'] - factor.loc[year_indices, 'rf_rate'],
            predictive=predictive.loc[year_indices]
        )
        model = factor_models.four_factor_model()
        # model = factor_models.conditional_four_factor_model()
        print(f"{name}, {year} : alpha = {round(model.params['const'], 2)}, p-val = {round(model.pvalues['const'], 2)}")

        df_alphas.at[name, year] = model.params['const'] # model.params.get('const', np.nan)
        df_p_values.at[name, year] = model.pvalues['const'] # model.pvalues.get('const', np.nan)

print(df_alphas)
print(df_p_values)

## Conversion to Excel Files:
df_alphas.to_excel("annualized_alphas_uncondi.xlsx", index=False)
df_p_values.to_excel("annualized_alphas_uncondi.xlsx", index=False)

## Excel files importation : 

# df_alphas = pd.read_excel('Data/annualized_alphas.xlsx')
# # df_alphas.index = pd.to_datetime(df_alphas.index, format=format)
# # df_alphas.sort_index(inplace=True)

# df_p_values = pd.read_excel('Data/annualized_alphas.xlsx')
# # df_p_values.index = pd.to_datetime(df_p_values.index, format=format)
# # df_p_values.sort_index(inplace=True)


######################################################## TEST FDR ANNUALISE ########################################################

# average_alphas_per_fund = df_alphas.mean(axis=1)
# average_p_values_per_fund = df_p_values.mean(axis=1)

# FDR_class = FDR(p_values=average_p_values_per_fund, alphas=average_alphas_per_fund, gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
# fdr_annualized = FDR_class.compute_fdr()
# proportion_annualized = FDR_class.compute_proportions(nb_simul=1000)

# print("Results for compute FDR (uncondi) : ", fdr_annualized)
# print("Results for compute FDR (condi) : ", fdr_annualized)


############################################################ GRAPHIQUES ############################################################

# ### Graph 1 : 

# zero_alphas_fund_per_year = (df_alphas == 0).sum(axis=0) # Nombre de zero-alpha funds observés 
# skilled_fund_per_year = (df_alphas > 0).sum(axis=0) # Nombre de skilled funds observés 
# unskilled_fund_per_year = (df_alphas < 0).sum(axis=0) # Nombre de unskilled funds observés 

# years_list = df_p_values.columns
# proportions_fund_per_year = np.full(shape=(len(years_list, 3)))
# year_i = 0
# for year in years_list :
#     FDR_year = FDR(p_values=df_p_values[year], alphas=df_alphas[year], gamma=0.5, lambda_threshold=0.6, pi0=0.75) # pi0=0.75
#     proportions = FDR_year.compute_proportions(nb_simul=100)
#     proportions_fund_per_year[year_i, 0] = proportions[0]
#     proportions_fund_per_year[year_i, 1] = proportions[1]
#     proportions_fund_per_year[year_i, 2] = proportions[2]
#     year_i += 1

# ### Graph 2 : 

# average_alphas_per_year = df_alphas.mean(axis=0)
# nb_funds_per_year = df_alphas.count(axis=0)


