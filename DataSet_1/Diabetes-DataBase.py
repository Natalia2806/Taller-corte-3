# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:59:47 2022

@author: Naty
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('diabetes.csv')

#Normalizar y limpiar la data

data.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], 
          axis= 1, inplace = True)

data.Age.replace(np.nan, 33, inplace=True)

ranges = [0, 8, 15, 18, 25, 40, 60, 100]

names = ['1', '2', '3', '4', '5', '6', '7']

data.Age = pd.cut(data.Age, ranges, labels=names)

data.dropna(axis=0,how='any', inplace=True)