import pandas as pd
import numpy as np
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore') # Suppresses warnings to avoid cluttering the output.

from data import factor, predictive, mutual_fund, weighted_portfolio

class FactorModels:
    """
    Class for constructing factor models in financial data analysis. 
    It supports both standard and conditional factor models using ordinary least squares (OLS).

    Attributes:
        exog (pd.DataFrame): Exogenous variables, typically factor loadings in a factor model.
        endog (pd.DataFrame): Endogenous variable, typically the returns of a portfolio or asset.
        predictive (pd.DataFrame, optional): Predictive variables used to modify the factor exposures conditionally, default is None.
    """
    
    def __init__(self, exog:pd.DataFrame, endog:pd.DataFrame, predictive:pd.DataFrame=None) -> None:
        """
        Initializes the FactorModels class with the provided dataframes.

        Parameters:
            exog (pd.DataFrame): The exogenous factors affecting the endogenous variable.
            endog (pd.DataFrame): The endogenous variable that the model tries to explain.
            predictive (pd.DataFrame, optional): Additional predictive variables for a conditional model.
        """
        self.exog = exog
        self.endog = endog
        self.predictive = predictive

    def four_factor_model(self) :  
        """
        Four factor model proposed by Carhart (1997)
        Constructs a standard four-factor model using OLS regression.

        Returns:
            pd.DataFrame: A data frame containing the coefficients and p-values of the regression model.
        """
        y = self.endog      
        X = self.exog.copy()
        X = sm.add_constant(X) # Add a constant term for the intercept (alpha)
        
        model = sm.OLS(y, X).fit() # Linear Regression OLS
        
        return pd.DataFrame({'Coeff': model.params, 'P-value': model.pvalues})
    
    def conditional_four_factor_model(self):   
        """
        Conditional four-factor model to account for time-varying exposure to the market porfolio (Fearson & Schadt (1996))
        Constructs a conditional four-factor model using OLS regression, where the factor loadings are adjusted by predictive variables.

        Raises:
            Exception: If predictive variables are not set when calling this method.

        Returns:
            pd.DataFrame: A data frame containing the coefficients and p-values of the regression model.
        """
        if self.predictive is None :
            raise("Select predictives variables for conditional factor model")  
        
        y = self.endog  
        X = self.exog.copy()
        
        # Add interaction terms to X_i : 
        for column in self.predictive.columns:
            for factor in self.exog.columns:
                X[f"{column}_{factor}"] = self.predictive[column] * X[factor]
        
        X = sm.add_constant(X) # Add a constant term for the intercept (alpha)
        
        model = sm.OLS(y, X).fit() # Linear Regression OLS
        
        return pd.DataFrame({'Coeff': model.params, 'P-value': model.pvalues})


# Résultats sur le portefeuille : 

Portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
print(Portfolio_results.four_factor_model())
print(Portfolio_results.conditional_four_factor_model())

# Remarque : la je ressort tous les coeff + les p-values qui vont avec 

# Résultats par fonds : 

nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
fund_index = 0 # Counter for funds

results = np.full((nb_funds, 5), fill_value=np.nan) 
# Col1 = alpha model 1, Col2 = p-value model 1, Col3 = alpha model 2, Col4 = p-value model 2, Col5 = nb dates for fund

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates = pd.Index(sorted(list(common_dates)))

    factor_models = FactorModels(exog = factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates, 'rdt'] - factor.loc[common_dates, 'rf_rate'] , 
                                 predictive = predictive.loc[common_dates])

    # OLS : 
    results_model_1 = factor_models.four_factor_model()
    results_model_2 = factor_models.conditional_four_factor_model()
    
    # Résults : 
    fund_names[fund_index] = name
    results[fund_index, 4] = len(common_dates)
    results[fund_index, 0] = results_model_1.loc['const', 'Coeff']
    results[fund_index, 1] = results_model_1.loc['const', 'P-value']
    results[fund_index, 2] = results_model_2.loc['const', 'Coeff']
    results[fund_index, 3] = results_model_2.loc['const', 'P-value']
    fund_index += 1

full_results = pd.DataFrame(results, index=fund_names, columns = ["alpha 1", "p-value 1", "alpha 2", "p-value 2","number of data"])
print(results)

# Remarque : la je me concentre que sur alpha + pvalue (pour les deux modèles)
    
    
