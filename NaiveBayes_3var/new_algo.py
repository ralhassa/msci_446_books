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

#find endocing values of Genre
"""
data1 = pd.read_csv('Dataset-3.csv')
data= pd.DataFrame({col: data1[col].astype('category').cat.codes for col in data1[['Genre']]}, index=data1.index)
labels = {col: {n: cat for n, cat in enumerate(data1[col].astype('category').cat.categories)}
    for col in data[['Genre']]}
print(labels)


# data encoding
"""
#getting the data

data1 = pd.read_csv('Dataset-3.csv')

data1['Genre']= pd.DataFrame({col: data1[col].astype('category').cat.codes for col in data1[['Genre']]}, index=data1.index)

#data

data = pd.DataFrame(data1)

#Naive Bayes
data=data[["rounded_book_count",
           "rounded_ratings_count",
           "Genre",
           "rounded_rating_2",
           "Class_New"
           ]].dropna(axis=0, how='any')


# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))
print(X_train, X_test)
# Instantiate the classifier
gnb = GaussianNB()
used_features =[
                "rounded_book_count",
                "rounded_ratings_count",
                "Genre",
                "rounded_rating_2"
                
                ]
# Train classifier
gnb.fit(
        X_train[used_features].values,
        X_train["Class_New"]
        )
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
              X_test.shape[0],
              (X_test["Class_New"] != y_pred).sum(),
              100*(1-(X_test["Class_New"] != y_pred).sum()/X_test.shape[0])
              ))

#resutls


mean_survival=np.mean(X_train["Class_New"])
mean_not_survival=1-mean_survival
print("Yes prob = {:03.2f}%, No prob = {:03.2f}%"
      .format(100*mean_survival,100*mean_not_survival))


# train on round_book_count

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
used_features =["rounded_book_count"]
y_pred = gnb.fit(X_train[used_features].values, X_train["Class_New"]).predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
              X_test.shape[0],
              (X_test["Class_New"] != y_pred).sum(),
              100*(1-(X_test["Class_New"] != y_pred).sum()/X_test.shape[0])
              ))
print("Std book count adapted = No: {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))
print("Std book count adapted= Yes: {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("Mean book count adapted = No: {:05.2f}".format(gnb.theta_[0][0]))
print("Mean book count adapted= Yes: {:05.2f}".format(gnb.theta_[1][0]))

# train on rounded_ratings_count

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
used_features =["rounded_ratings_count"]
y_pred = gnb.fit(X_train[used_features].values, X_train["Class_New"]).predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
              X_test.shape[0],
              (X_test["Class_New"] != y_pred).sum(),
              100*(1-(X_test["Class_New"] != y_pred).sum()/X_test.shape[0])
              ))
print("Std ratings count adapted = No: {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))
print("Std ratings count adapted= Yes: {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("Mean ratings count adapted = No: {:05.2f}".format(gnb.theta_[0][0]))
print("Mean ratings count adapted= Yes: {:05.2f}".format(gnb.theta_[1][0]))

# train on round_rating

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
used_features =["rounded_rating_2"]
y_pred = gnb.fit(X_train[used_features].values, X_train["Class_New"]).predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
              X_test.shape[0],
              (X_test["Class_New"] != y_pred).sum(),
              100*(1-(X_test["Class_New"] != y_pred).sum()/X_test.shape[0])
              ))
print("Std rating adapted = No: {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))
print("Std rating count adapted= Yes: {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("Mean rating count adapted = No: {:05.2f}".format(gnb.theta_[0][0]))
print("Mean rating count adapted= Yes: {:05.2f}".format(gnb.theta_[1][0]))
