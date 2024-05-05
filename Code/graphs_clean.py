import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from data import factor, predictive, mutual_fund, FUND_RETURN
from regression import FactorModels

class Graphs:
    """
    A class to handle the entire process of loading data, performing regression analysis,
    and visualizing results for mutual fund performance evaluation using factor models.

    Attributes:
        factor (pd.DataFrame): A DataFrame containing factor data with date as index.
        predictive (pd.DataFrame): A DataFrame containing predictive variables with date as index.
        mutual_fund (pd.DataFrame): A DataFrame containing mutual fund data.
        full_results (pd.DataFrame): A DataFrame to store the results of the regression analysis.
    
    Methods:
        prepare_data(): Processes data to compute regressions for each fund.
        tstat_graph(tstat): Plots the density of t-statistics across all funds.
        tstat_graph_by_category(tstat, category): Plots the density of t-statistics by alpha categories.
    """
    def __init__(self, factor, predictive, mutual_fund):
        """
        Constructs all the necessary attributes for the Graphs object.

        Parameters:
            factor (pd.DataFrame): DataFrame containing factor returns with required columns like 'mkt_rf', 'smb', 'hml', 'mom'.
            predictive (pd.DataFrame): DataFrame containing predictive variables.
            mutual_fund (pd.DataFrame): DataFrame containing mutual fund data with columns 'fundname' and 'return'.
        """

        self.factor = factor
        self.predictive = predictive
        self.mutual_fund = mutual_fund
        self.full_results = None
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares the data for analysis by performing regression analysis on mutual fund returns
        adjusted by risk-free rates using factor models. This method updates the `full_results` attribute
        with the alpha, t-stat, and category for each fund.
        """

        nb_funds = len(self.mutual_fund['fundname'].unique())
        fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
        results = np.full((nb_funds, 2), fill_value=np.nan)
        fund_index = 0

        for name, fund in self.mutual_fund.groupby('fundname'):
            common_dates = set(self.factor.index).intersection(set(fund.index)).intersection(set(self.predictive.index))
            common_dates = pd.Index(sorted(list(common_dates)))
            factor_models = FactorModels(exog=self.factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                         endog=fund.loc[common_dates, FUND_RETURN] - self.factor.loc[common_dates, 'rf_rate'], 
                                         predictive=self.predictive.loc[common_dates])
            four_factor = factor_models.four_factor_model()
            fund_names[fund_index] = name
            results[fund_index, 0] = four_factor.params['const']
            results[fund_index, 1] = four_factor.tvalues[0]
            fund_index += 1

        self.full_results = pd.DataFrame(results, index=fund_names, columns=['alpha', 't-stat'])
        self.full_results['Category'] = np.where(self.full_results['alpha'] > 1, 'pos', 
                                                 np.where(self.full_results['alpha'] < -1, 'neg', 'zero'))

    def tstat_graph(self, tstat):
        """
        Plots the density of t-statistics for the specified model across all funds.

        Parameters:
            tstat (str): Column name of the t-statistic in the `full_results` DataFrame.
        """

        plt.figure(figsize=(10, 5))
        data_norm = (self.full_results[tstat] - self.full_results[tstat].mean()) / self.full_results[tstat].std()
        kde = sns.kdeplot(data_norm, label="t-stat", color="slategrey")
        kde_fill = kde.get_lines()[-1]
        plt.fill_between(kde_fill.get_xdata(), kde_fill.get_ydata(), 
                         where=(kde_fill.get_xdata() < -1.65) | (kde_fill.get_xdata() > 1.65), color="lightgray", alpha=0.5)
        plt.title("Density of t-statistics")
        plt.xlabel("t-statistic")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def tstat_graph_by_category(self, tstat, category):
        """
        Plots the density of t-statistics by alpha categories for the specified model.

        Parameters:
            tstat (str): Column name of the t-statistic in the `full_results` DataFrame.
            category (str): Column name of the fund categories based on alpha in the `full_results` DataFrame.
        """
        plt.figure(figsize=(10, 5))
        categories = {'neg': -2.5, 'zero': 0, 'pos': 3}
        for key in categories:
            values = self.full_results[self.full_results[category] == key][tstat]
            norm = (values - values.mean()) / values.std()
            legend = "Unskilled Funds" if key == 'neg' else "Skilled Funds" if key == "pos" else "Zero-alpha Funds"
            kde = sns.kdeplot(norm + categories[key], label=legend, color="tomato" if key == 'neg' else "slategrey" if key == 'zero' else "royalblue")
            # f"{key.capitalize()} funds"
            if key== 'zero' :
                zero_fill = kde.get_lines()[-1]        
                plt.fill_between(zero_fill.get_xdata(), zero_fill.get_ydata(), 
                     where=(zero_fill.get_xdata() < -1.65) | (zero_fill.get_xdata() > 1.65), color="lightgray", alpha=0.5)
        
        plt.title("Density of t-statistics by alpha categories")
        plt.xlabel("t-statistic")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def pvalue_histogram(self, proportions, means, std_dev, sample_size=4320):
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

graphs = Graphs(factor, predictive, mutual_fund)
graphs.tstat_graph("t-stat")
graphs.tstat_graph_by_category("t-stat", "Category")

graphs.pvalue_histogram(graphs.full_results['Category'].value_counts() / len(graphs.mutual_fund), [0, -2.5, 3], 1)
