import numpy as np

class FDR:
    def __init__(self, p_values:np.ndarray, alphas:np.ndarray, gamma:float, lambda_threshold:float, pi0:float=None) -> None:
        self.p_values = p_values
        self.gamma = gamma
        self.alphas = alphas
        self.lambda_threshold = lambda_threshold
        if pi0 is None:
            self.pi0 = self.estimate_pi0()
        else :
            self.pi0 = pi0
        
    def estimate_pi0(self):
        """ Estime la proportion de fonds à alpha nul dans la population (π₀)."""
        # W = Funds with p-values exceeding lambda_treshold = number of zero-alpha_funds
        W = self.p_values[self.p_values > self.lambda_threshold]
        pi0 = len(W) / len(self.p_values) / (1.0 - self.lambda_threshold)
        return pi0

    def compute_fdr(self):
        """
        Compute the False Discovery Rate (FDR) using the given significance level gamma.
        Returns overall FDR, FDR+ (positive FDR), and FDR- (negative FDR).
        """
        
        # Compute for all data:
        significant_prop = np.sum(self.p_values < self.gamma) / len(self.p_values)
        fdr = self.pi0 * self.gamma / significant_prop if significant_prop > 0 else 0
        
        expected_false_positives = self.pi0 * self.gamma / 2
        
        # Compute for the negative side
        neg_p_values = self.p_values[self.alphas < 0]
        negatives_prop = np.sum(neg_p_values < self.gamma) / len(neg_p_values) if len(neg_p_values) > 0 else 0
        fdr_neg = (expected_false_positives / negatives_prop) if negatives_prop > 0 else 0
        
        # Compute for the positive side
        pos_p_values = self.p_values[self.alphas > 0]
        positives_prop = np.sum(pos_p_values < self.gamma) / len(pos_p_values) if len(pos_p_values) > 0 else 0
        fdr_pos = (expected_false_positives / positives_prop) if positives_prop > 0 else 0

        return round(fdr, 2), round(fdr_neg, 2), round(fdr_pos, 2)


        
        
    def compute_proportions(self, nb_simul) :
        """
        Compute the null, negative, and positive proportions of p-values based on a bootstrap method,
        considering the sign of alphas. This function also calculates the overall proportion of these categories.

        Parameters:
            nbsimul (int): Number of bootstrap simulations to perform.

        Returns:
            tuple: Contains four floats representing the null proportion, proportion of significant negative alphas,
               proportion of significant positive alphas, and the overall proportion of these categories.
        """
        nb_funds = len(self.p_values) 
        treshold_test = np.arange(0.40, 1.0, 0.05)  # Array of thresholds to test for finding the optimal p-value threshold
        nb_test = len(treshold_test)  
        
        null_proportion = np.zeros(nb_test)  # Array to store the proportion of null hypothesis tests that pass each threshold
        # Calculate proportion of null hypothesis (p-values >= threshold) for each threshold
        for i in range(nb_test):
            nb_zero_alpha = np.sum(self.p_values >= treshold_test[i])  # Count p-values that are above the current threshold 
            null_proportion[i] = (nb_zero_alpha / nb_funds) / (1 - treshold_test[i])  # Adjusted proportion of null hypotheses
        min_null_prop = np.min(null_proportion)  # Minimum proportion across all thresholds
        
        null_proportion_boot = np.zeros((nb_test, nb_simul))  # Array to store bootstrap results
        # Perform bootstrap simulations
        for j in range(nb_simul):
            p_values_sample = np.random.choice(self.p_values, size=nb_funds, replace=True) 
            for i in range(nb_test):
                nb_zero_alpha_boot = np.sum(p_values_sample >= treshold_test[i])   # Count p-values that are above the current threshold
                null_proportion_boot[i, j] = nb_zero_alpha_boot / ((1 - treshold_test[i]) * nb_funds)  # Adjusted proportion of null hypotheses

        mse = np.mean(np.square(null_proportion_boot - min_null_prop), axis=1)  # Mean squared error of bootstrap proportions from the minimum
        index = np.argmin(mse)  # Index of the threshold with the lowest MSE
        optimal_threshold = treshold_test[index]  # Optimal threshold based on MSE
        print(f"gamma optim = {optimal_threshold}")
        zero_alpha_prop = null_proportion[index]  # Proportion of null hypothesis for optimal threshold
        zero_alpha_prop = np.clip(zero_alpha_prop, 0, 1)  # Ensure proportion is within valid range (0 to 1)

        # Calculate proportions of significant negative alphas using optimal threshold
        neg_p_values = self.p_values[self.alphas < 0]  # Subset p-values for negative alphas
        negative_prop = np.sum(neg_p_values < optimal_threshold) / nb_funds  # Proportion of negative alphas below optimal threshold
        negative_prop = max(negative_prop - zero_alpha_prop * optimal_threshold / 2, 0)  # Adjusted proportion 
        
        # Calculate proportions of significant positive alphas using optimal threshold
        pos_p_values = self.p_values[self.alphas > 0]  # Subset p-values for positive alphas
        positive_prop = np.sum(pos_p_values < optimal_threshold) / nb_funds  # Proportion of positive alphas below optimal threshold
        positive_prop = max(positive_prop - zero_alpha_prop * optimal_threshold / 2, 0)  # Adjusted proportion

        return zero_alpha_prop, negative_prop, positive_prop, np.sum([zero_alpha_prop, positive_prop, negative_prop])
        
    # def compute_proportions_2(self) :
    #     """ Calcule les proportions de fonds incompétents, zéro-alpha et compétents.
    #     Returns: proportion de fonds incompétents, zéro-alpha, et compétents."""

    #     zero_alpha = (np.sum(self.p_values >= self.gamma) / len(self.p_values)) / ((1 - self.gamma))
    #     unskilled = max(np.sum((self.alphas < 0) & (self.p_values < self.gamma)) / len(self.p_values) - (zero_alpha * self.gamma / 2), 0)
    #     skilled = max(np.sum((self.alphas > 0) & (self.p_values < self.gamma)) / len(self.p_values) - (zero_alpha * self.gamma / 2), 0)
    #     # unskilled = np.mean((self.alphas < 0) & (self.p_values < self.gamma)) 
    #     # skilled = np.mean((self.alphas > 0) & (self.p_values < self.gamma))
    #     # zero_alpha = 1 - (unskilled + skilled)

    #     return round(zero_alpha*100, 2), round(unskilled*100, 2), round(skilled*100, 2)


#######################################################################################################################

from data import mutual_fund, factor, predictive
from regression import FactorModels
import pandas as pd

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
    # conditional_four_factor = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 0] = four_factor.params['const']
    results[fund_index, 1] = four_factor.pvalues['const']
    # results[fund_index, 2] = conditional_four_factor.params['const']
    # results[fund_index, 3] = conditional_four_factor.pvalues['const']
    fund_index += 1

# full_results = pd.DataFrame(results, index=fund_names, columns = ['alpha 1', 'p-value 1', 'alpha 2', 'p-value 2'])

# TEST FDR : 
# test = FDR(p_values=results[:, 1], alphas=results[:,0], lambda_treshold=0.6, significiance_level=0.05)
test = FDR(p_values=results[:, 1], alphas=results[:,0], gamma=0.4, lambda_threshold=0.7)
# fdr = test.compute_fdr()
proportion = test.compute_proportions(nb_simul=100)
# proportion_2 = test.compute_proportions_2()

# print("Results for compute FDR", fdr)
print("Results for compute proportions", proportion)
# print("Results for compute proportions 2", proportion_2)

"""
    J'ai pas voulu trop touché au code du gentil monsieur mais je suis vraiment pas sure pour le compute_fdr
    Si c'est exprimé en pourcentage ca a du sens mais suis pas sure que ca soit le cas :(
    Pour le computation_proportion, les résultats m'on l'air un peu louche mais quand je fouille dans la fonction tout me parait bien ... 
    (j'ai juste changer tous les noms des variables pour mieux comprendre)
"""
