# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:54:35 2018

@author: Melanie
"""

import numpy as np
import pandas as pd

from rulefit import RuleFit

boston_data = pd.read_csv("prism_numeric.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

rf = RuleFit()
rf.fit(X, y, feature_names=features)
rf.predict(X)
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values("support", ascending=False)

print(rules)