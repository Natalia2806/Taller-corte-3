# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:59:47 2022

@author: Naty
"""


import numpy as np
import pandas as pd

##Librerias para graficar la matriz de confusión
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

##Librerias para validación cruzada
from sklearn.model_selection import KFold

##Librerias de metricas para matriz de confusión
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('weatherAUS.csv')

#Normalizar y limpiar la data

data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainToday.value_counts()

data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data.RainTomorrow.value_counts()

data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)


#Dividir la data en train y test 
data_train = data[:38767]
data_test = data[38767:]

x = np.array(data_train.drop(['RainTomorrow'], axis=1))
y = np.array(data_test.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], axis=1))
y_test_out = np.array(data_test.RainTomorrow)

## REGRESIÓN LOGÍSTICA

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entrenamiento el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y): 
    logreg.fit(x[train], y[train]) 
    scores_train_train = logreg.score(x[train], y[train]) 
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) #Graficar la matriz
plt.figure(figsize = (6, 6)) #Dimesiones de la matriz
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1}')


#--------------------------------------------------------------------------------------------

## MAQUINA DE SOPORTE VECTORIAL

svc = SVC(gamma='auto')

# Entrenamiento el modelo
svc.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10) #5 hace referencia a las cinco partes en las que dividirá el data

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): #Split ayuda a dividir el data y los resultados los guarda en train y test, luego se realiza el for en dichos resultados
    svc.fit(x[train], y[train]) #Entrenamiento del data
    scores_train_train = svc.score(x[train], y[train]) #Resultados de entrenamiento
    scores_test_train = svc.score(x[test], y[test]) #Resultados de entrenamiento
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('Regresión Logística MAQUINA DE SOPORTE VECTORIAL')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) #Graficar la matriz
plt.figure(figsize = (6, 6)) #Dimesiones de la matriz
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1}')

#---------------------------------------------------------------------------------------------------

# RANDOM FOREST

random_forest = RandomForestClassifier()

# Entrenar el modelo

random_forest.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test, y_test)}')

# Accuracy de Entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {random_forest.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test_out, y_test_out)}')

# RANDOM FOREST CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    random_forest.fit(x[train], y[train]) 
    scores_train_train = random_forest.score(x[train], y[train]) 
    scores_test_train = random_forest.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('Regresión Logística RANDOM FOREST')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {random_forest.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1}')

#-----------------------------------------------------------------------------------------------------

## ARBOL DE DECISIÓN

arbol = DecisionTreeClassifier()

# Entrenamiento el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    arbol.fit(x[train], y[train]) 
    scores_train_train = arbol.score(x[train], y[train]) 
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('ARBOL DE DECISIÓN')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1}')


#-------------------------------------------------------------------------------------------------------



# K-Nearest neighbors

# Seleccionar un modelo

kneighbors = KNeighborsClassifier()

# Entrenar el modelo

kneighbors.fit(x_train, y_train)

# Metricas

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {kneighbors.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test_out, y_test_out)}')

# K-Nearest neighbors CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    kneighbors.fit(x[train], y[train]) 
    scores_train_train = kneighbors.score(x[train], y[train]) 
    scores_test_train = kneighbors.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {kneighbors.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1}')



