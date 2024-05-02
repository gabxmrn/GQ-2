# Data Files

## Original Model

### File 1 - exogeneous_variables

01.1975 to 12.2006 monthly - 384 data points

rf_rate :
    used the CSRP 1-month risk free rate as a proxy

mkt_ptf :
    03.1980 to 12.2006 : total return of NYSE/AMEX market value weighted portfolio (Mrkt portfolio in %)

smb :
    Size portfolio from Kenneth French's Website
    Average of the Monthly Value Weighted Returns for the portfolios (average done on Low 10, Dec 2 to 9, High 10 | could have done the average on Low 20, Qnt 2 to 4, Hi 20 => same number of companies in the sample but different averages, minimal difference)

hml :
    Book to market portfolio from Kenneth French's Website
    Average of the Monthly Value Weighted Returns for the portfolios (average done on Low 10, Dec 2 to 9, High 10 | could have done the average on Low 20, Qnt 2 to 4, Hi 20 => same number of companies in the sample but different averages, minimal difference)

mom :
    Momentum portfolio from Kenneth French's Website
    Average of the Monthly Value Weighted Returns for the 10 constructed portfolios

1M_Tbill :
    Used the CSRP US Treasuries and Inflation Indexes 30 days yield
    Different from the risk free rate (computed by calculating the return of the bill value, it is not a rate)

div_yield_mkt :
    03.1980 to 12.2006 : income return of NYSE/AMEX market value weighted portfolio (Mrkt portfolio in %)

term_spread :
    Difference between the 10Y treasuries and 3M T-Bill
    Both series come from the FRED database, source US government
    Average done on each month

default_spread :
    Difference between Moody's Baa-rated and Moody's Aaa-rated corporate bonds
    Data from the FRED

### File 2 - variables_expliquées

01.1975 to 12.2006 monthly
