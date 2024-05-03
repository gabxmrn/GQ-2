import pandas as pd
import numpy as np
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore') # Suppresses warnings to avoid cluttering the output.

from data import factor, predictive, mutual_fund

######################################################### FOUR FACTOR MODEL #########################################################

def four_factor_model(exog:pd.DataFrame, endog:pd.DataFrame) :
        """
                Four factor model proposed by Carhart (1997)
                Perform linear regression of mutual fund returns on risk factors.
        """
        
        nb_funds = len(endog['fundname'].unique()) # Total number of funds 
        fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
        results = np.full((nb_funds, 3), fill_value=np.nan) # Col1 = alpha, Col2 = p-value, Col3 = nb dates for fund
        fund_index = 0 # Counter for funds
        
        for name, fund in endog.groupby('fundname'):
                common_dates = set(exog.index).intersection(set(fund.index))
                common_dates = pd.Index(sorted(list(common_dates)))
                fund_names[fund_index] = name
                results[fund_index, 2] = len(common_dates)
                
                if len(common_dates) < 2 :
                        fund_index += 1
                        continue
                
                X_i = exog.loc[common_dates, ['mkt_ptf', 'smb', 'hml', 'mom']]  
                X_i = sm.add_constant(X_i) # Add a constant term for the intercept (alpha)
                y_i = fund['rdt'].loc[common_dates] - exog['rf_rate'].loc[common_dates] # Excess Returns 
                
                # Régression linéaire OLS
                model = sm.OLS(y_i, X_i).fit()
                
                results[fund_index, 0] = model.params['const'] # alpha
                results[fund_index, 1] = model.pvalues['const'] # p-value
                fund_index += 1
                
        return pd.DataFrame(results, index=fund_names, columns = ["alpha", "p-value", "number of data"])


################################################### CONDITIONAL FOUR FACTOR MODEL ###################################################

def conditional_four_factor_model(exog: pd.DataFrame, endog: pd.DataFrame, predictive: pd.DataFrame):
        """
                Conditional four-factor model to account for time-varying exposure to the market porfolio (Fearson & Schadt (1996))
                Fit a conditional four-factor model with interactions between predictive variables and factors.
        """
    
        nb_funds = len(endog['fundname'].unique()) # Total number of funds 
        fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
        results = np.full((nb_funds, 3), fill_value=np.nan) # Col1 = alpha, Col2 = p-value, Col3 = nb dates for fund
        fund_index = 0 # Counter for funds
        
        for name, fund in endog.groupby('fundname'):
                common_dates = set(exog.index).intersection(set(fund.index)).intersection(set(predictive.index))
                common_dates = pd.Index(sorted(list(common_dates)))
                fund_names[fund_index] = name
                results[fund_index, 2] = len(common_dates)

                if len(common_dates) < 2:
                        fund_index += 1
                        continue

                X_i = exog.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']]
                predictive_i = predictive.loc[common_dates]
        
                # Add interaction terms to X_i : 
                for column in predictive_i.columns:
                        for factor in ['mkt_rf', 'smb', 'hml', 'mom']:
                                X_i[f"{column}_{factor}"] = predictive_i[column] * X_i[factor]
        
                X_i = sm.add_constant(X_i) # Add a constant term for the intercept (alpha)
                y_i = fund.loc[common_dates, 'rdt'] - exog.loc[common_dates, 'rf_rate']  # Excess returns
        
                # Régression linéaire OLS
                model = sm.OLS(y_i, X_i).fit()
                
                results[fund_index, 0] = model.params['const'] # alpha
                results[fund_index, 1] = model.pvalues['const'] # p-value
                fund_index += 1
                
        return pd.DataFrame(results, index=fund_names, columns = ["alpha", "p-value", "number of data"])



model_1 = four_factor_model(exog=factor, endog=mutual_fund)
print(model_1)
model_2 = conditional_four_factor_model(exog=factor, endog=mutual_fund, predictive=predictive)
print(model_2)


"""
Remarques : 
        - J'ai convertie toute les données exogènes en trimestrielles pour que tout coincide 
        - Je fais une regression par fonds, pour chacun d'eux je filtre les facteurs sur les dates dispo 
        (donc les résulats des regressions pour chaque fonds ne concernent pas les mêmes dates)
        - Pour la variable r_{t,i} (y) je suis pas sure de moi (excess return ou pas excess return)  
                # "the month t excess return of fund i over the risk-free rate (proxied by the monthly 30-day T-bill beginning-of-month yield)" ?????  
                En attend, modele 1 recalcule un excess return à partir du RFR, pas le modele 2 (comme ca on voit a peut pres la diff) 
        - Pour les variables mkt_ptf et div_yield_mkt j'ai bien vu le (Mrkt portfolio in %) dans la methodo 
                mais je sais pas comment je dois le prendre en compte dans le code 
        - Pour le modele 2, dans les results les coeffs existent mais j'ai souvent des nan pour les autres variables (stat)
        
        PS : C'est long a tourner 
"""