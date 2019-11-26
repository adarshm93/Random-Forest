# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:35:42 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np
# Reading the Data #################
company = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/random forest/Company_Data(1).csv")
company.head()
company.columns

#converting into categorical data
np.median(company["Sales"]) #7.49
company["sales"]="<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"

#converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
company["ShelveLoc"]=lb.fit_transform(company["ShelveLoc"])
company["Urban"]=lb.fit_transform(company["Urban"])
company["US"]=lb.fit_transform(company["US"])
company["sales"]=lb.fit_transform(company["sales"])

company.drop(["Sales"],inplace=True,axis=1)

colnames = list(company.columns)
predictors = colnames[:10]
target = colnames[10]

X = company[predictors]
Y = company[target]

# Splitting data into training and testing data set

from sklearn.model_selection import train_test_split
train,test = train_test_split(company,test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=150,criterion="entropy")
rf.fit(train[predictors],train[target])
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 10 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.77
rf.predict(X)


pred_train=rf.predict(train[predictors])
pred_test = rf.predict(test[predictors])
pd.Series(pred_test).value_counts()
pd.crosstab(test[target],pred_test)
pd.crosstab(train[target],pred_train)

# Accuracy = train
np.mean(train.sales == rf.predict(train[predictors])) #1.0

# Accuracy = Test
np.mean(pred_test==test.sales) # 80.00
