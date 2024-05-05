import pandas as pd
import numpy as np

from data import factor, predictive, mutual_fund, weighted_portfolio, FUND_RETURN
from regression import FactorModels
from computations import computationfdr, computationproportions


######################################################## REGRESSIONS SUR LE PORTEFEUILLE ########################################################

# Portfolio_results = FactorModels(exog=factor[['mkt_rf', 'smb', 'hml', 'mom']], endog=weighted_portfolio["Excess Returns"], predictive=predictive)
# print(Portfolio_results.four_factor_model())
# print(Portfolio_results.conditional_four_factor_model())

# Remarque : la je ressort tous les coeff + les p-values qsui vont avec 
# LE CODE EST PLUS D'ACTUALITE AVEC LES CHANGEMENTS QUE J'AI FAIT AUJOURD'HUI

###################################################################### FDR ######################################################################

nb_funds = len(mutual_fund['fundname'].unique()) # Total number of funds 
fund_names = np.full(nb_funds, fill_value=np.nan, dtype='object')
fund_index = 0 # Counter for funds

results = np.full((nb_funds, 4), fill_value=np.nan) 
# Col1 = alpha model 1, Col2 = p-value model 1, Col3 = alpha model 2, Col4 = p-value model 2

for name, fund in mutual_fund.groupby('fundname'):
    # Dates management : 
    common_dates = set(factor.index).intersection(set(fund.index)).intersection(set(predictive.index))
    common_dates = pd.Index(sorted(list(common_dates)))

    # OLS : 
    factor_models = FactorModels(exog = factor.loc[common_dates, ['mkt_rf', 'smb', 'hml', 'mom']], 
                                 endog = fund.loc[common_dates, FUND_RETURN] - factor.loc[common_dates, 'rf_rate'] , 
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
pval = results[:, 1]
test_fdr = computationfdr(pvec=pval, pnul=0.16, threshold=0.1) # pnul=0.16
test_proportion = computationproportions(pvec=pval, nbsimul=1000)

print(test_fdr)
print(test_proportion)

"""
1) Selection des dates : OK
2) Vecteur des p-values : OK
3) Test FDR : OK (résultats bizarres ? on a un truc infini) -> j'ai check tout ça et effectvement bizarre, mais je pense que c'est 
pcq il doit y avoir un division par 0. J'ai modifié pour gérer des cas, et maintenant on a 0 au lieu de inf (est-ce que le résultat fait sens??)
4) Tester les graphs : OK (aucun changement avec nouveau momentum) --> rajout d'une classe Graphs où on pourra rajouter dees choses au fur et à mesure :)

"""