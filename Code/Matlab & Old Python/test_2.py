import pandas as pd
import statsmodels.api as sm
import numpy as np

from data import factor, mutual_fund, FUND_RETURN, common_dates

def calculate_quarterly_alpha(mutual_fund, factor):
    # Initialize the DataFrame to store alpha values
    alphas = pd.DataFrame()

    for name, group in mutual_fund.groupby('fundname'):
        for date in group.index:
            if date in factor.index:
                fund_data = group[group.index == date]
                factor_data = factor.loc[[date]]  # Ensure factor data is a DataFrame

                if not fund_data.empty and not factor_data.empty:
                    # Get returns and factors for the date
                    fund_returns = fund_data[FUND_RETURN].values - factor_data['rf_rate'].values
                    exog = sm.add_constant(factor_data[['mkt_rf', 'smb', 'hml', 'mom']])  # Adding constant for intercept

                    # Ensure there's more than one observation to run OLS
                    if len(fund_returns) > 0 and exog.shape[0] == len(fund_returns):
                        # OLS regression
                        model = sm.OLS(fund_returns, exog, missing='drop').fit()
                        alpha = model.params.get('const')

                        # Store the alpha value
                        alphas.loc[date, name] = alpha

    return alphas

# Compute the quarterly alphas
quarterly_alphas = calculate_quarterly_alpha(mutual_fund, factor)

# Check the result
print(quarterly_alphas.head())
