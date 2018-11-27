import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
#getting the data 

data1 = pd.read_csv('Dataset_3var.csv')


# data encoding 


data1['Genre']= pd.DataFrame({col: data1[col].astype('category').cat.codes for col in data1[['Genre']]}, index=data1.index)

print(data1)

labels = {col: {n: cat for n, cat in enumerate(data1[col].astype('category').cat.categories)} 
     for col in data1[['Genre']]}
     
print(labels)
     
#/ data encoding 

data = pd.DataFrame(data1)
 
print(data)

#code
data=data[["adapted",
           "rating",
           "Genre",
           "ratings_count"
           ]].dropna(axis=0, how='any')


# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))

# Instantiate the classifier
gnb = GaussianNB()
used_features =[
                 "rating",
                 "Genre",
                "ratings_count"
                
                ]
# Train classifier
gnb.fit(
        X_train[used_features].values,
        X_train["adapted"]
        )
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
              X_test.shape[0],
              (X_test["adapted"] != y_pred).sum(),
              100*(1-(X_test["adapted"] != y_pred).sum()/X_test.shape[0])
              ))

#resutls

"""
mean_survival=np.mean(X_train["adapted"])
mean_not_survival=1-mean_survival
print("Survival prob = {:03.2f}%, Not survival prob = {:03.2f}%"
      .format(100*mean_survival,100*mean_not_survival))
"""
