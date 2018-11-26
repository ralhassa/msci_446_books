# adapted from https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

data = pd.read_csv('Goodreads_10k-list_apriori_classYES.csv', encoding='cp1252')

records = []
for i in range(0, 500):
    records.append([str(data.values[i, j]) for j in range(0, 3)])

association_rules = apriori(records, min_support=0.0001, min_confidence=0.6, min_lift=2, min_length=2)
association_results = list(association_rules)
print(len(association_results))
count = len(association_results)
for i in range(0, count):
    print(association_results[i])

# store_data = pd.read_csv('store_data.csv', header=None)
# records = []
# for i in range(0, 7501):
#     records.append([str(store_data.values[i, j]) for j in range(0, 20)])
#
# association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
# association_results = list(association_rules)
# print(len(association_results))
# print(association_results[0])
