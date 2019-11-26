# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:25:26 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np
data = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/random forest/Fraud_check(1).csv")
data.head()
data.columns

#converting into categorical data
data["taxincome"]="<=30000"
data.loc[data["Taxable.Income"]>=30000,"taxincome"]="Good"
data.loc[data["Taxable.Income"]<=30000,"taxincome"]="Risky"

data.drop(["Taxable.Income"],inplace=True,axis=1)

#converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data["Undergrad"]=lb.fit_transform(data["Undergrad"])
data["Marital.Status"]=lb.fit_transform(data["Marital.Status"])
data["Urban"]=lb.fit_transform(data["Urban"])

colnames = list(data.columns)
predictors = colnames[:5]
target = colnames[5]

X = data[predictors]
Y = data[target]

# Splitting data into training and testing data set

from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=200,criterion="entropy")
rf.fit(train[predictors],train[target])
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 10 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.747
rf.predict(X)


pred_train=rf.predict(train[predictors])
pred_test = rf.predict(test[predictors])
pd.Series(pred_test).value_counts()
pd.crosstab(test[target],pred_test)
pd.crosstab(train[target],pred_train)

# Accuracy = train
np.mean(train.taxincome == rf.predict(train[predictors])) #1.0

# Accuracy = Test
np.mean(pred_test==test.taxincome) # 74.00
