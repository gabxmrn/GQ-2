import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stationnarization import Stationnarity_Test

FUND_RETURN = "mret"
FUND_DATE = "caldt"
FUND_NAME = "fund_name"

########################################################## DATA IMPORTATION #########################################################

"""
        Load and preprocess exogeneous variables (market factors and predictive variables)
        and endogeneous variables (mutual funds data). 
"""

# Period selection : 
startdate = "1980-01"
enddate = "2006-12"

# Full Exogeneous Variables
exogeneous_variables = pd.read_excel("Data/exogeneous_variables.xlsx", index_col='Dates')
exogeneous_variables.index = pd.to_datetime(exogeneous_variables.index)
exogeneous_variables.dropna(inplace=True)
exogeneous_variables.sort_index()
exogeneous_variables.index = exogeneous_variables.index.strftime('%Y-%m') 

# Factor variables for both factor models (Carhart &  Ferson/Schadt)
factor = exogeneous_variables.loc[startdate:enddate, ['rf_rate', 'mkt_ptf', 'smb', 'hml', 'mom']] 
factor['mkt_rf'] = factor['mkt_ptf'] - factor['rf_rate'] 

# Predictive variables of the conditionnal 4 factors model :
predictive = exogeneous_variables.loc[startdate:enddate, ['1M_Tbill_yield', 'div_yield_mkt_ptf', 'term_spread', 'default_spread']]
for column in predictive.columns:
        predictive[column] -= predictive[column].mean() # value at end of month t - mean on the period
predictive = predictive.shift(1)  # Data shift to represent z_{t-1}
predictive.dropna(inplace=True)

# Dates management : 
common_dates = pd.Index(sorted(set(factor.index).intersection(set(predictive.index))))
factor = factor.loc[common_dates, :]

# Mutual funds data
mutual_fund = pd.read_csv("Data/crsp_mutual_funds_1975_2021.csv")
# mutual_fund = mutual_fund.groupby(FUND_NAME).filter(lambda x: x[FUND_RETURN].notnull().rolling(window=59).count().max() >= 59)
mutual_fund[FUND_DATE] = pd.to_datetime(mutual_fund[FUND_DATE])
mutual_fund[FUND_DATE] = mutual_fund[FUND_DATE].apply(lambda x: x.strftime('%Y-%m'))
mutual_fund.replace([np.inf, -np.inf], np.nan, inplace=True)
mutual_fund.dropna(inplace=True)
mutual_fund = mutual_fund[mutual_fund[FUND_DATE].isin(common_dates)] # Sames dates as exogeneous variables
mutual_fund[FUND_RETURN] = mutual_fund[FUND_RETURN] * 100
mutual_fund = mutual_fund.sort_values(by=[FUND_NAME, FUND_DATE])

# # 60 dates consécutives
# def has_60_consecutive_months(group):
#     # Ensure the date column is sorted
#     group = group.sort_values(FUND_DATE)
#     expected_next_dates = group[FUND_DATE] + pd.DateOffset(months=1)
#     actual_next_dates = group[FUND_DATE].shift(-1)
#     is_consecutive = (actual_next_dates == expected_next_dates).fillna(False)
    
#     return is_consecutive.rolling(window=59, min_periods=1).sum().max() >= 59 

# # Group by fund and apply the function
# mutual_fund[FUND_DATE] = pd.to_datetime(mutual_fund[FUND_DATE])
# mutual_fund = mutual_fund.groupby(FUND_NAME).filter(has_60_consecutive_months)
# mutual_fund = mutual_fund.sort_values(by=[FUND_NAME, FUND_DATE])

####################################################### DATA STATIONNARIZATION ######################################################

Stationnarity_Test_factor = Stationnarity_Test(factor, 10, 0.05) 
# print(Stationnarity_Test_factor.isStationnary)
# Non-stationnary -> rf_rate, mom
factor = Stationnarity_Test_factor.get_stationnarized_data()
Stationnarity_Test_factor = Stationnarity_Test(factor, 10, 0.05) 
# print(Stationnarity_Test_factor.isStationnary)
# print(Stationnarity_Test_factor.get_results())

Stationnarity_Test_predictive = Stationnarity_Test(predictive, 10, 0.05) 
# print(Stationnarity_Test_predictive.isStationnary)
# Non-stationnary -> 1M_Tbill_yield,  div_yield_mkt_ptf,  default_spread
predictive = Stationnarity_Test_predictive.get_stationnarized_data() 
Stationnarity_Test_predictive = Stationnarity_Test(predictive, 10, 0.05) 
# print(Stationnarity_Test_predictive.isStationnary)
# print(Stationnarity_Test_predictive.get_results())

Stationnarity_Test_fund = Stationnarity_Test(mutual_fund, 10, 0.05) 
# print(Stationnarity_Test_fund.isStationnary)
# Non-stationnary -> none 
# print(Stationnarity_Test_fund.get_results())


# ######################################################### PORTFOLIO CREATION ########################################################

mutual_fund_returns = mutual_fund.pivot_table(index=FUND_NAME, columns=FUND_DATE, values=FUND_RETURN, aggfunc='first')
mutual_fund.set_index(FUND_DATE, drop=True, inplace=True)

nb_funds_per_dates = mutual_fund_returns.notnull().sum()
weighted_averages = mutual_fund_returns.sum() * (1 / nb_funds_per_dates)

weighted_portfolio = pd.DataFrame(weighted_averages, columns=['Returns'])
weighted_portfolio["Nb funds"] = nb_funds_per_dates


# ########################################################## DATES MANAGEMENT #########################################################

common_dates = pd.Index(sorted(set(weighted_portfolio.index).intersection(set(factor.index)).intersection(set(predictive.index))))
weighted_portfolio = weighted_portfolio.loc[common_dates, :]
factor = factor.loc[common_dates, :]
predictive = predictive.loc[common_dates, :]

weighted_portfolio["Excess Returns"] = weighted_portfolio["Returns"] - factor["rf_rate"] # Excess returns over risk free rate

# # ########################################################################################################
# # fig, ax1 = plt.subplots(figsize=(12, 6))

# # color = 'tab:blue'
# # ax1.set_xlabel('fDate')
# # ax1.set_ylabel('Returns', color=color)
# # ax1.plot(weighted_portfolio.index, weighted_portfolio['Returns'], color=color)
# # ax1.tick_params(axis='y', labelcolor=color)
# # ax1.grid(True)

# # ax2 = ax1.twinx()
# # color = 'tab:red'
# # ax2.set_ylabel('Nb funds', color=color)  
# # ax2.plot(weighted_portfolio.index, weighted_portfolio['Nb funds'], color=color, linestyle='--')
# # ax2.tick_params(axis='y', labelcolor=color)

# # plt.title('Returns')
# # plt.tight_layout()
# # plt.show()
# # ########################################################################################################
