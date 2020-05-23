# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:25:47 2020

@author: jackn
"""
import pandas as pd
import numpy as np

'''using pandas, import the train, test, and gender_submission csv files '''
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_gendersub = pd.read_csv("gender_submission.csv")

