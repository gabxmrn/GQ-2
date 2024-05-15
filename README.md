# Outils pour la Gestion Quantitative - Semestre 2
## **Implémentation d'un papier de recherche : *False Discoveries in Mutual Fund Performance: Measuring Luck in Estimated Alphas* Barras, Scaillet and Wermers (2010)**
### BHAUGEERUTTY, KIRCH, MORIN


### Création d'un environnement virtuel
```bash
python3 -m venv venv
source venv/Scripts/activate # sur Windows
source venv/bin/activate # ou sinon
```
### Installation des modules prérequis 
```bash
pip install -r requirements.txt
```

### Clonage du repository Github
Vous pouvez accéder à tous nos travaux mis à jour en rentrant la commande suivante dans le terminal, dans le dossier où vous souhaitez initialiser le projet.
```bash
git clone https://github.com/gabxmrn/GQ-2.git
```


### Présentation technique du projet
Le projet se compose de 3 parties principales : le fichier principal (`main.py`), le fichier contenant le code des régressions afin d'estimer les $\alpha$ (`regression.py`), et le fichier contenant le code du False Discovery Rate (FDR), le calcul des proportions et des biais (`computations.py`). Les autres fichiers servent à l'importation et traitement des données (nettoyage, stationarisation), et aux graphiques afin de visualiser nos résultats.
#### Le `data.py`
Si nous voulons changer les dates du sample, il suffit de modifer les lignes ci-dessous du fichier `data.py` (au format "YYYY-MM"), en sachant que la date de la dernière observation est à décembre 2023.

```python
# Period selection : 
startdate = "1980-03"
enddate = "2023-12"
```

De plus, ayant réalisé les tests sur deux bases de données (CSRP et Thomson), si nous souhaitons visualiser les résultats avec l'autre base de données, il suffit de modifier le path du mutual funds data (ligne 46 du fichier `data.py`).
```python
mutual_fund = pd.read_csv("Data/mutual_funds_1975_2023.csv")
### ou
mutual_fund = pd.read_csv("Data/crsp_mutual_funds_1975_2021.zip")
```
#### Le `main.py`
Ce fichier principal sert à appeler les fonctions qui se trouvent dans d'autres fichiers. 

Pour commencer, il faut importer fonctions, classes et données nécessaires à l'obtention des résultats : 
```python
from data import factor, predictive, mutual_fund, common_dates, weighted_portfolio, FUND_RETURN, FUND_DATE, FUND_NAME
from regression import FactorModels
from graphs import tstat_graph, tstat_graph_by_category, pvalue_histogram
from computations import FDR
```

Les tableaux de l'impact de la chance sur la performance est visualisable avec la fonction `table_impact_of_luck`. Ainsi si nous voulons voir ces résultats, il suffit de faire tourner la commande suivante, en spécifiant les différents seuils de significativité, le nombre de simulations, la proportion de $\pi_0$, et le seuil du $\lambda$ pour lesquels nous souhaitons avoir les résultats :

```python
significance_levels = [0.05, 0.10, 0.15, 0.20]
pval_uncondi, alphas_uncondi, t_stat_uncondi = results[:, 1], results[:, 0], results[:, 2]
impact_of_luck_uncondi = table_impact_of_luck(regression=full_results, 
                                              significance_levels=significance_levels, 
                                              model="uncondi", 
                                              lambda_treshold=0.6, 
                                              nb_simul=1000,
                                              pi0=0.3)
print(impact_of_luck_uncondi)
```
