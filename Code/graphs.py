import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess


##############################################################################################################################
### Graphiques 1 : Cross-sectional t-statistic distribution

def tstat_graph(data:pd.DataFrame, tstat:str) :
    plt.figure(figsize=(10, 5))
    
    # KDE plot
    data_norm = (data[tstat] - data[tstat].mean()) / data[tstat].std()
    kde = sns.kdeplot(data_norm, label="t-stat", color="slategrey")
    kde_fill = kde.get_lines()[-1]
    # Fill the area between
    plt.fill_between(kde_fill.get_xdata(), kde_fill.get_ydata(), 
                     where=(kde_fill.get_xdata() < -1.65) | (kde_fill.get_xdata() > 1.65), color="lightgray", alpha=0.5)
    
    plt.title("Cross-sectional t-statistic distribution")
    plt.xlabel("t-statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

##############################################################################################################################

##############################################################################################################################
### Graph 2 : Individual fund t-statistics distribution

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
    
    plt.title("Individual fund t-statistics distribution")
    plt.xlabel("t-statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show() 

##############################################################################################################################

##############################################################################################################################
### Graph 3 : Density Histogram of p-values

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

##############################################################################################################################
    
    
##############################################################################################################################
### Graph 4 : Proportion of unskilled & skilled funds  
    
def graph_proportions(proportions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    color='black'
    
    # Tracé des courbes sans marqueurs et avec des styles de ligne distincts
    ax.plot(proportions.index, proportions['zero_alpha funds'], label='Zero Alpha Funds', color=color)  # Ligne pleine
    ax.plot(proportions.index, proportions['skilled funds'], linestyle='--', label='Skilled Funds', color=color)  # Ligne pointillée
    ax.plot(proportions.index, proportions['unskilled funds'], linestyle='-.', label='Unskilled Funds', color=color)  # Ligne pointillée-tiretée
    
    # Configuration des axes et du graphique
    ax.set_title('Proportion of unskilled & skilled funds')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Funds')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=len(proportions.index)//2))
    ax.legend()
    
    plt.show()

##############################################################################################################################

   
##############################################################################################################################
### Graph 5 : Total number of funds & Average alphas per years

def graph_alphas(timeline: np.ndarray, alphas: np.ndarray, nb_funds: np.ndarray):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Configuration de l'axe pour les alphas
    color = 'black'  # Couleur noire pour les lignes
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual average alphas', color=color)
    smoothed_alphas = lowess(alphas, timeline, frac=0.3)[:, 1]  # Smoothing the alphas
    ax1.plot(timeline, smoothed_alphas, linestyle='--', color=color, label='Alphas') 
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=len(timeline)//2))
    
    # Configuration de l'axe secondaire pour le nombre de fonds
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total number of Funds', color=color)
    ax2.plot(timeline, nb_funds, linestyle='-', color=color, label='Funds') 
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Total number of funds & Average alphas')
    fig.tight_layout()
    plt.show()
    
##############################################################################################################################