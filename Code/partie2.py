import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from partie1 import full_results

"""
    - Graphique des p-value
    (étape avant le FDR ??)
"""

np.random.seed(0) 

# Parameters
sample_size = 8525  # Number of funds 
counts = full_results['Category 1'].value_counts()
proportions = counts / counts.sum() # Proportion of each categories of alpha
print(proportions)
means = [0, -2.5, 3] 
std_dev = 1 

t_stats = np.concatenate([
    np.random.normal(mean, std_dev, int(sample_size * proportion))
    for mean, proportion in zip(means, proportions)
])

p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

plt.hist(p_values, bins=10, range=(0, 1), color='gray', edgecolor='black')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('Histogram of p-values')
plt.show()