import numpy as np
from scipy.stats import t, nct

class FDR:
    """
    Class designed to handle the computation of False Discovery Rates (FDR).
    This class allows users to estimate the proportion of true null hypotheses (pi0) and calculate FDR 
    for both overall and directional hypotheses (positive and negative effects).

    Attributes:
        p_values (np.ndarray): Array of p-values obtained from factor models.
        alphas (np.ndarray): Corresponding alpha values for each p-value.
        gamma (float): Significance level threshold for determining if a finding is statistically significant.
        lambda_threshold (float): Threshold used to estimate the proportion of true null hypotheses (pi0).
        pi0 (float, optional): Estimated proportion of true null hypotheses; simulated if not provided.
    """
    
    def __init__(self, p_values:np.ndarray, alphas:np.ndarray, gamma:float, lambda_threshold:float, pi0:float=None) -> None:
        """ Initializes the FDR class. """
        self.p_values = p_values
        self.gamma = gamma
        self.alphas = alphas
        self.lambda_threshold = lambda_threshold
        if pi0 is None:
            self.pi0 = self.estimate_pi0()
        else :
            self.pi0 = pi0
        
    def estimate_pi0(self):
        """
        Estimates pi0 (proportion of true null hypotheses) based on the lambda_threshold method, 
            assuming a uniform distribution over the null p-values.
        
        Returns:
            float: The estimated proportion of true null hypotheses (pi0).
        """
        
        W = self.p_values[self.p_values > self.lambda_threshold] # number of zero-alpha_funds
        pi0 = len(W) / len(self.p_values) / (1.0 - self.lambda_threshold)
        return pi0

    def compute_fdr(self):        
        """
        Computes the False Discovery Rate (FDR) based on gamma and the estimated pi0, for both overall and directional hypotheses.

        Returns:
            tuple: Contains the overall FDR, FDR for negative alphas, and FDR for positive alphas.
        """
        
        # Compute for overall FDR:
        significant_prop = np.sum(self.p_values < self.gamma) / len(self.p_values)
        fdr = self.pi0 * self.gamma / significant_prop if significant_prop > 0 else 0
        
        expected_false_positives = self.pi0 * self.gamma / 2 # Expected number of false positives under the null
        
        # Compute for the negative side
        neg_p_values = self.p_values[self.alphas < 0]
        negatives_prop = np.sum(neg_p_values < self.gamma) / len(neg_p_values) if len(neg_p_values) > 0 else 0
        # negatives_prop = np.sum(neg_p_values < self.gamma) / significant_prop if significant_prop > 0 else 0
        fdr_neg = (expected_false_positives / negatives_prop) if negatives_prop > 0 else 0
        
        # Compute for the positive side
        pos_p_values = self.p_values[self.alphas > 0]
        positives_prop = np.sum(pos_p_values < self.gamma) / len(pos_p_values) if len(pos_p_values) > 0 else 0
        # positives_prop = np.sum(pos_p_values < self.gamma) / significant_prop if significant_prop > 0 else 0
        fdr_pos = (expected_false_positives / positives_prop) if positives_prop > 0 else 0

        return round(fdr, 4), round(fdr_neg, 4), round(fdr_pos, 4)
        
    def compute_proportions(self, nb_simul) :        
        """
        Computes proportions of significant findings under various p-value thresholds using bootstrap simulations, 
            and determines the optimal threshold for significance based on minimizing the mean squared error (MSE) 
            between simulated proportions and the observed minimum proportion.

        Parameters:
            nb_simul (int): Number of bootstrap simulations to perform for estimating stability and accuracy of p-value thresholds.

        Returns:
            tuple: Contains proportions of zero alpha significance, negative and positive significant alphas, and the total of these proportions.
        """
        
        nb_funds = len(self.p_values) 
        treshold_test = np.arange(0.05, 1.0, 0.05) 
        nb_test = len(treshold_test)  
      
        # Calculate proportion of null hypothesis (p-values >= threshold) for each threshold
        null_proportion = np.zeros(nb_test)
        for i in range(nb_test):
            nb_zero_alpha = np.sum(self.p_values >= treshold_test[i])  
            null_proportion[i] = (nb_zero_alpha / nb_funds) / (1 - treshold_test[i])  
        min_null_prop = np.min(null_proportion) 
        
        # Perform bootstrap simulations and calculate proportion of null hypothesis for each threshold
        null_proportion_boot = np.zeros((nb_test, nb_simul)) 
        for j in range(nb_simul):
            p_values_sample = np.random.choice(self.p_values, size=nb_funds, replace=True) 
            for i in range(nb_test):
                nb_zero_alpha_boot = np.sum(p_values_sample >= treshold_test[i]) 
                null_proportion_boot[i, j] = nb_zero_alpha_boot / ((1 - treshold_test[i]) * nb_funds)

        mse = np.mean(np.square(null_proportion_boot - min_null_prop), axis=1)  
        index = np.argmin(mse) 
        optimal_threshold = treshold_test[index]  # Optimal threshold based on MSE
        zero_alpha_prop = null_proportion[index]  
        zero_alpha_prop = np.clip(zero_alpha_prop, 0, 1) 

        # Calculate proportions of significant negative alphas using optimal threshold
        neg_p_values = self.p_values[self.alphas < 0] 
        neg_prop = np.sum(neg_p_values < optimal_threshold) / nb_funds 
        neg_prop = max(neg_prop - zero_alpha_prop * optimal_threshold / 2, 0)  
        
        # Calculate proportions of significant positive alphas using optimal threshold
        pos_p_values = self.p_values[self.alphas > 0] 
        pos_prop = np.sum(pos_p_values < optimal_threshold) / nb_funds 
        pos_prop = max(pos_prop - zero_alpha_prop * optimal_threshold / 2, 0) 

        return round(zero_alpha_prop, 4), round(neg_prop, 4), round(pos_prop, 4), round(np.sum([zero_alpha_prop, pos_prop, neg_prop]), 4)


    def compute_bias(self, t_stats, T):
        """
        Calculate the delta(lambda) from t-statistics using noncentral t-distribution.

        Parameters:
            T (int): Number of observations (used as degrees of freedom).

        Returns:
            float: Delta(lambda) indicating misclassification probability.
        """
        noncentrality = np.mean(np.abs(t_stats))

        lower_bound = t.ppf(self.lambda_threshold / 2, df=T-1)
        upper_bound = t.ppf(1 - (self.lambda_threshold / 2), df=T-1)

        F_nc = nct.cdf(upper_bound, df=T-1, nc=noncentrality) - nct.cdf(lower_bound, df=T-1, nc=noncentrality)

        # Compute delta(lambda)
        delta_lambda = F_nc / (1 - self.lambda_threshold)

        return round(delta_lambda, 2)
    
    
    def compute_bias_simple(self, expected_pi0):
        """
        Calculate the delta(lambda) from t-statistics using noncentral t-distribution.

        Parameters:
            T (int): Number of observations (used as degrees of freedom).

        Returns:
            float: Delta(lambda) indicating misclassification probability.
        """
        E_pi_alpha = 1 - expected_pi0
        _, neg_prop, pos_prop, _ = self.compute_proportions(nb_simul=1000)
        delta_lambda = ((neg_prop + pos_prop) - E_pi_alpha) / (neg_prop + pos_prop)
 
        return round(delta_lambda, 2)

