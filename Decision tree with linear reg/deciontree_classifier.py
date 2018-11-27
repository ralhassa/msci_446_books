# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:08:56 2018
Decision tree classification
@author: Melanie
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
dataset = pd.read_csv("prism_dataset_nogenre.csv")  
dataset.shape  
dataset.head()  
X = dataset.drop('Class', axis=1)  
y = dataset['Class'] 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10) 
#train and make predictions
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)   
y_pred = classifier.predict(X_test) 
#evaluate algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  