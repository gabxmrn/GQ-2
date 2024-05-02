import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


###################################################### STATIONNARIZATION CLASS ######################################################

class Stationnarity_Test:
    """
    Class for performing stationarity tests on time series data.

    Attributes:
    - df: DataFrame, the time series to be tested.
    - max_lags: int, the maximum number of lags to include in the Augmented Dickey-Fuller (ADF) test.
    - threshold: float, significance threshold for the ADF test.

    Usage:
    - Create an instance of the class by providing the time series, maximum number of lags, and threshold.
    - Call the full_stationnarity_test method to perform tests on all columns.
    - Retrieve results using get_results and obtain the stationarized time series with get_stationnarized_data.
    - Stationnarize time series using get_stationnarized_data method. 
    """
    
    def __init__(self, df:pd.DataFrame, max_lags:int, threshold:float) -> None :
        """
        Initializes the class with the time series, maximum number of lags, and significance threshold.

        Inputs :
            - df: DataFrame, the time series to be tested.
            - max_lags: int, the maximum number of lags to include in the ADF test.
            - threshold: float, significance threshold for the ADF test.
        """
        self.df = df
        self.max_lags = max_lags
        self.threshold = threshold
        self.isStationnary = pd.DataFrame(index=['isStationnary'])
        self.stationnarity = pd.DataFrame(index=['ct', 'c', 'n'], columns=df.columns)
        self.trend = pd.DataFrame(index=['Trend Value', 'Significativity'], columns=df.columns)
        self.constant = pd.DataFrame(index=['Constant value ct', 'Significativity ct', 'Constant value c', 'Significativity c'], columns=df.columns)
        self.full_stationnarity_test()


    def full_stationnarity_test(self) -> None :
        """
        Performs stationarity tests for each column in the time series.
        """
        for col in self.df.columns:
            if self.df[col].dtype == "object" :
                continue
        #     print(self.df[col].dtype)
            self._sequential_strategy(self.df[col])


    def get_results(self) -> pd.DataFrame :
        """
        Returns the results of stationarity, trend, and constant tests as a DataFrame.
        """
        return pd.concat([self.stationnarity, self.trend, self.constant], axis=0)


    def get_stationnarized_data(self) -> pd.DataFrame :
        """
        Returns a stationarized version of the time series.

        Outputs :
            - DataFrame, stationarized version of the time series.
        """
        
        results = pd.DataFrame(index=self.df.index[1:])
        for col in self.df.columns:
            if not self.isStationnary[col].iloc[0]:
                results[col] = np.diff(self.df[col])
            else:
                results[col] = self.df[col].iloc[1:]
        return results


    def _sequential_strategy(self, df:pd.core.series.Series) -> None :
        """
        Performs specific stationarity tests and updates the results (stationnarity and variables significativity tests).

        Inputs :
            - df: Series, a column of the time series DataFrame.
        """
        
        self.stationnarity.at["ct", df.name] = self._dickey_fuller(df, "ct")
        self.trend.at["Trend Value", df.name], self.trend.at["Significativity", df.name], self.constant.at["Constant value ct", df.name], self.constant.at["Significativity ct", df.name] = self._linear_regression(df, "ct")

        if not self.trend.at["Significativity", df.name]:
            self.stationnarity.at["c", df.name] = self._dickey_fuller(df, "c")
            self.constant.at["Constant value c", df.name], self.constant.at["Significativity c", df.name] = self._linear_regression(df, "c")

            if not self.constant.at["Significativity c", df.name]:
                self.stationnarity.at["n", df.name] = self._dickey_fuller(df, "n")
                if self.stationnarity.at["n", df.name]:
                    self.isStationnary[df.name] = True
                else:
                    self.isStationnary[df.name] = False

            else:
                if self.stationnarity.at["c", df.name]:
                    self.isStationnary[df.name] = True
                else:
                    self.isStationnary[df.name] = False

        else:
            if self.stationnarity.at["ct", df.name]:
                self.isStationnary[df.name] = True
            else:
                self.isStationnary[df.name] = False


    def _dickey_fuller(self, df:pd.core.series.Series, model_type:str) -> bool :
        """
        Conducts the Augmented Dickey-Fuller (ADF) test to assess stationarity.

        Inputs :
            - df: Series, a column of the time series DataFrame.
            - model_type: str, the type of model for the ADF test ('ct', 'c' or 'n')

        Output :
            - bool, True if the series is stationary, False otherwise.
        """
        ADF = adfuller(df, self.max_lags, model_type, "AIC")
        return ADF[1] <= self.threshold


    def _linear_regression(self, Y:pd.core.series.Series, model_type:str) -> list :
        """
        Performs linear regression to assess trend and constant values.

        Inputs :
            - Y: Series, a column of the time series DataFrame.
            - model_type: str, the type of model for regression ('ct', 'c' or 'n')

        Output :
            - list, [constant value, significance of constant, trend value, significance of trend] if applicable.
        """
        n = len(Y)
        x = np.arange(1, n + 1)
        if model_type == "ct":
            X = np.column_stack((np.ones(n), x, x**2))
            model = sm.OLS(Y, X)
            results = model.fit()
            ####################################################################################################################################################################
            # print(f"cool = {results.params.iloc[1]}, bad = {results.pvalues['x1']}") 
            # return [results.params.iloc[0], results.params.iloc[0] <= self.threshold, results.params.iloc[1], results.params.iloc[1] <= self.threshold]
            return [results.params.iloc[0], results.pvalues["const"] <= self.threshold, results.params.iloc[1], results.pvalues["x1"] <= self.threshold]
        elif model_type == "c":
            X = np.column_stack((np.ones(n), x))
            model = sm.OLS(Y, X)
            results = model.fit()
            ####################################################################################################################################################################
            # print(f"cool = {results.params.iloc[0]}, bad = {results.pvalues['const']}") 
            # return [results.params.iloc[0], results.params.iloc[0] <= self.threshold]
            return [results.params.iloc[0], results.pvalues["const"] <= self.threshold]
        else:
            pass
        
    
    def kpss_check(self) -> pd.DataFrame :
        """
        Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity
        on each column of the DataFrame stored in the object.

        Parameters:
           - self: The instance of the class containing the DataFrame.
           - threshold: The significance level to determine stationarity.

        Returns:
           - pd.DataFrame: A DataFrame with True or False values indicating whether each column 
             is stationary (True) or not (False) based on the KPSS test.
        """
        results = []
        for col in self.df.columns:
            kpss_result = kpss(self.df[col])
            results.append(kpss_result.iloc[1] >= self.threshold)
        return pd.DataFrame([results], columns=self.df.columns, index=['KPSS test'])
        


########################################################## DATA IMPORTATION #########################################################

"""
        Load and preprocess exogeneous variables (market factors and predictive variables)
        and endogeneous variables (mutual funds data). 
"""

# Full Exogeneous Variables
exogeneous_variables = pd.read_excel("Data/exogeneous_variables.xlsx", index_col='Dates')
exogeneous_variables.index = pd.to_datetime(exogeneous_variables.index) 
exogeneous_variables = exogeneous_variables.resample('QE').mean() # Resample data to quarterly to match funds data
exogeneous_variables.dropna(inplace=True)
exogeneous_variables.index = exogeneous_variables.index.strftime('%Y-%m') 
start_date = "1980-03"

# Factor variables for both factor models (Carhart &  Ferson/Schadt)
factor = exogeneous_variables.loc[start_date:, ['rf_rate', 'mkt_ptf', 'smb', 'hml', 'mom']]
factor['mkt_rf'] = factor['mkt_ptf'] - factor['rf_rate'] # Excess returns over risk free rate ????? 

# Predictive variables of the conditionnal 4 factors model :
predictive = exogeneous_variables.loc[start_date:, ['1M_Tbill_yield', 'div_yield_mkt_ptf', 'term_spread', 'default_spread']]
for column in predictive.columns:
        predictive[column] -= predictive[column].mean() # value at end of month t - mean on the period
predictive = predictive.shift(1)  # Data shift to represent z_{t-1}
predictive.dropna(inplace=True)

# Mutual funds data
mutual_fund = pd.read_csv("Data/mutual_funds.csv", dtype={'cusip': str})
mutual_fund['fdate'] = pd.to_datetime(mutual_fund['fdate'])
mutual_fund.set_index('fdate', drop=True, inplace=True)
mutual_fund.index = mutual_fund.index.strftime('%Y-%m')
mutual_fund.replace([np.inf, -np.inf], np.nan, inplace=True)
mutual_fund.dropna(inplace=True)


####################################################### DATA STATIONNARIZATION ######################################################

Stationnarity_Test_factor = Stationnarity_Test(factor, 10, 0.05)
# print(Stationnarity_Test_factor.isStationnary)
# Non stationnaire -> rf_rate, mom
factor = Stationnarity_Test_factor.get_stationnarized_data()
Stationnarity_Test_factor = Stationnarity_Test(factor, 10, 0.05)
# print(Stationnarity_Test_factor.isStationnary)
# print(Stationnarity_Test_factor.get_results())

Stationnarity_Test_predictive = Stationnarity_Test(predictive, 10, 0.05)
# print(Stationnarity_Test_predictive.isStationnary)
# Non stationnaire -> 1M_Tbill_yield,  div_yield_mkt_ptf,  default_spread
predictive = Stationnarity_Test_predictive.get_stationnarized_data()
Stationnarity_Test_predictive = Stationnarity_Test(predictive, 10, 0.05)
# print(Stationnarity_Test_predictive.isStationnary)
# print(Stationnarity_Test_predictive.get_results())

Stationnarity_Test_fund = Stationnarity_Test(mutual_fund, 10, 0.05)
print(Stationnarity_Test_fund.isStationnary)
# Non stationnaire -> nada :) 
# print(Stationnarity_Test_fund.get_results())