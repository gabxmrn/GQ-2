import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore') # Suppresses warnings to avoid cluttering the output.


class FactorModels:
    """
    Class for constructing factor models in financial data analysis. 
    It supports both standard and conditional factor models using ordinary least squares (OLS).

    Attributes:
        exog (pd.DataFrame): Exogenous variables, typically factor loadings in a factor model.
        endog (pd.DataFrame): Endogenous variable, typically the returns of a portfolio or asset.
        predictive (pd.DataFrame, optional): Predictive variables used to modify the factor exposures conditionally, default is None.
        risk_free_rate (pd.DataFrame, optional): Risk Free Rate to compute Excess returns of fund, default is None. 
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
        """
        
        y = self.endog
        X = self.exog.copy()
        X = sm.add_constant(X) # Add a constant term for the intercept (alpha)

        # cov_type='HAC',cov_kwds={'maxlags':1}
        # -> Dit dans l'article mais renvoie des résultats un peu plus louches (graphiquement)
        # "To compute each fund t-statistic, we use the Newey and West (1987) heteroskedasticity and autocorrelation consistent estimator of the standard deviation"
        return sm.OLS(y, X).fit(cov_type='HAC',cov_kwds={'maxlags':1}) # Linear Regression OLS
    
    def conditional_four_factor_model(self):   
        """
        Conditional four-factor model to account for time-varying exposure to the market porfolio (Fearson & Schadt (1996))
        Constructs a conditional four-factor model using OLS regression, where the factor loadings are adjusted by predictive variables.

        Raises:
            Exception: If predictive variables are not set when calling this method.
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

        return sm.OLS(y, X).fit(cov_type='HAC',cov_kwds={'maxlags':1}) # Linear Regression OLS
    
    
    
    
    
