# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:25:47 2020

@author: jackn
"""
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


'''using pandas, import the train, test, and gender_submission csv files '''
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_gendersub = pd.read_csv("gender_submission.csv")

combine = [df_train,df_test]

#print the first row of data
print(df_train.iloc[0])

'''
from the visual analysis of the data, the following variables 
can be considered categorical: 
*    Survived
*    Sex
*    Embarked
Ordinal:
*    Pclass
Continuous:
*    Age
*    Fare
Discrete:
*    SibSp
*    Parch
Mixed:
*   Ticket
'''
#describe function provides a very quick statistical analytical view of the data
#the count quickly reveals that Age is not fully populated and requires addressing.
train_desc = df_train.describe
print(train_desc)

#view the average survival rate grouped by Passenger class to understand broad % of survival across PClass
Pclass_view = df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Pclass_view)

#view average gender % who survived
Gender_view = df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Gender_view)

#view of average survival per Parch
Parch_view = df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Parch_view)








