import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


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

##############################################################################################################################

##############################################################################################################################
### Graphiques 3 : Histogramme des p-values

def pvalue_histogram(proportions, means, std_dev, sample_size):
    np.random.seed(0)
    t_stats = np.concatenate([
        np.random.normal(mean, std_dev, int(sample_size * proportion))
        for mean, proportion in zip(means, proportions)
    ])
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    counts, bins = np.histogram(p_values, bins=10, range=(0, 1), density=True)
    counts = counts / 10
    plt.bar(bins[:-1], counts, width=np.diff(bins), color='gray', edgecolor='black')
    plt.xlabel('p-value')
    plt.ylabel('Density')
    plt.ylim(0, 0.4)
    plt.title('Density Histogram of p-values')
    plt.show()