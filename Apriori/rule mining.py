# adapted from https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

import pandas as pd
from apyori import apriori

data = pd.read_csv('Dataset_apriori_yes.csv', encoding="utf8")

records = []
for i in range(0, 1139):
    records.append([str(data.values[i, j]) for j in range(0, 7)])

association_rules = apriori(records, min_support=0.01, min_confidence=0.5,  min_lift=1.5, min_length=2)
association_results = list(association_rules)
print(len(association_results))
# count = len(association_results)
# for i in range(0, count):
#     print(association_results[i])

for item in association_results:

    # printing out all orderedStatistics
    for i in range(0, len(item[2])):
        # print(item[2])
        print("Base: " + str(item[2][i][0]))
        print("Add: " + str(item[2][i][1]))
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][i][2]))
        print("Lift: " + str(item[2][i][3]))
        print()

    print("=====================================")

