import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from data import factor, predictive, mutual_fund
from regression import FactorModels

"""
    - Regression sur tous les fonds 
    - Graphique des t-stats 
    - Histogramme des p-values 
"""

nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
fund_index = 0 # Counter for funds

results = np.full((nb_funds, 4), fill_value=np.nan) 
# Col1 = alpha model 1, Col2 = t-stat model 1, Col3 = alpha model 2, Col4 = t-stat model 2

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates = pd.Index(sorted(list(common_dates)))

    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates, 'rdt'] - factor.loc[common_dates, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates])
    four_factor = factor_models.four_factor_model()
    # conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 0] = four_factor.params['const']
    results[fund_index, 1] = four_factor.tvalues[0]
    # results[fund_index, 2] = conditional_four_factor.params['const']
    # results[fund_index, 3] = conditional_four_factor.tvalues[0]
    fund_index += 1

full_results = pd.DataFrame(results, index=fund_names, columns = ['alpha 1', 't-stat 1', 'alpha 2', 't-stat 2'])

# Create categories based on 'alpha'
full_results['Category 1'] = np.where(full_results['alpha 1'] > 1, 'pos', np.where(full_results['alpha 1'] < -1, 'neg', 'zero'))
# full_results['Category 2'] = np.where(full_results['alpha 2'] > 1, 'pos', np.where(full_results['alpha 1'] < -1, 'neg', 'zero'))


##############################################################################################################################
### Graphiques 1 : transversale (tous les fonds quel que soit leur alpha) pour le modèle simple à 4 facteurs 

def tstat_graph(data:pd.DataFrame, tstat:str) :
    plt.figure(figsize=(10, 5))
    
    # KDE plot
    data_norm = (data[tstat] - data[tstat].mean()) / data[tstat].std()
    kde = sns.kdeplot(data_norm, label="t-stat", color="slategrey")
    kde_fill = kde.get_lines()[-1]
    # Fill the area between
    plt.fill_between(kde_fill.get_xdata(), kde_fill.get_ydata(), 
                     where=(kde_fill.get_xdata() < -1.65) | (kde_fill.get_xdata() > 1.65), color="lightgray", alpha=0.5)
    
    plt.title("Density of t-statistics")
    plt.xlabel("t-statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

tstat_graph(full_results, "t-stat 1") # four factor model (comme dans l'article)
# tstat_graph(full_results, "t-stat 2") # conditionnal four factor model (nul)

##############################################################################################################################


##############################################################################################################################
### Graphiques 2 : 3 catégories de alpha (neg, zero, pos) 

def tstat_graph_by_category(data: pd.DataFrame, tstat: str, category: str):
    plt.figure(figsize=(10, 5))
    
    categories = {'neg': -2.5, 'zero': 0, 'pos': 3}
    
    # KDE plot for 'Négatif'
    negative_values = data[data[category] == 'neg'][tstat]
    negative_norm = (negative_values - negative_values.mean()) / negative_values.std()
    sns.kdeplot(negative_norm + categories['neg'], label="Unskilled funds", color="tomato")
    
    # KDE plot for 'Null'
    null_values = data[data[category] == 'zero'][tstat]
    null_norm = (null_values - null_values.mean()) / null_values.std()
    kde_null = sns.kdeplot(null_norm + categories['zero'], label="Zero-alpha funds", color="slategrey")
    kde_null_fill = kde_null.get_lines()[-1]
    # kde_null_fill.set_color("lightgray")
    plt.fill_between(kde_null_fill.get_xdata(), kde_null_fill.get_ydata(), 
                     where=(kde_null_fill.get_xdata() < -1.65) | (kde_null_fill.get_xdata() > 1.65), color="lightgray", alpha=0.5)
    
    # KDE plot for 'Positif'
    positive_values = data[data[category] == 'pos'][tstat]
    positive_norm = (positive_values - positive_values.mean()) / positive_values.std()
    sns.kdeplot(positive_norm + categories['pos'], label="Skilled funds", color="royalblue")
    
    plt.title("Density of t-statistics by alpha categories")
    plt.xlabel("t-statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

tstat_graph_by_category(full_results, "t-stat 1", "Category 1")
# tstat_graph_by_category(full_results, "t-stat 2", "Category 2")   

##############################################################################################################################


##############################################################################################################################
### Graphiques 3 : Histogramme des p-values

np.random.seed(0) 

# Parameters
#### Paramétrer les proportions sur les 5 dernières années du sample 
sample_size = 8525  # Number of funds 
counts = full_results['Category 1'].value_counts()
proportions = counts / counts.sum() # Proportion of each categories of alpha
means = [0, -2.5, 3] 
std_dev = 1 

t_stats = np.concatenate([
    np.random.normal(mean, std_dev, int(sample_size * proportion))
    for mean, proportion in zip(means, proportions)
])

p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

counts, bins = np.histogram(p_values, bins=10, range=(0, 1), density=True)
counts = counts / 10  # Mise à l'échelle
plt.bar(bins[:-1], counts, width=np.diff(bins), color='gray', edgecolor='black')
plt.xlabel('p-value')
plt.ylabel('Density')
plt.ylim(0, 0.4)  # Ajuster l'échelle
plt.title('Density Histogram of p-values')
plt.show()

##############################################################################################################################