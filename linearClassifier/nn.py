# -*- coding: utf-8 -*-
"""
# ML05: Neural Network on iris by Numpy
# https://merscliche.medium.com/ml05-8771620a2023
# @author: Morton Kuo (2020/10/30)
"""

#%% (1) Input  

import numpy as np
import pandas as pd


df = pd.read_csv('IRIS.csv')
df.drop(df.index[0])
# print(df)
x = df.iloc[0:100,[0, 1, 2, 3]].values
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', 0, 1)



#%% (2) Split the dataset into train & test
x_train = np.empty((80, 4))
x_test = np.empty((20, 4))
y_train = np.empty(80)
y_test = np.empty(20)

x_train[:40],x_train[40:] = x[:40],x[50:90]
x_test[:10],x_test[10:] = x[40:50],x[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]


#%% (3) Define functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x, weights, bias):
    return sigmoid(np.dot(x, weights)+bias)

def update(x, y_train, weights, bias, learning_rate): 
    #ACIVATION
    y_pred = activation(x, weights, bias)
    # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    sigmoid_derivative = y_pred * (1- y_pred)
    # print(y_pred)
    #PARTIAL DERIVATIVE
    a = (y_pred - y_train) * sigmoid_derivative
    # print(a)
    for i in range(4):
        weights[i] -= learning_rate * 1/float(len(y)) * np.sum(a*x[:,i])
    bias -= learning_rate * 1/float(len(y))*np.sum(a)
    return weights, bias


#%% (4) Training
weights = np.random.rand(4)
bias = np.random.rand(1)
eta = 0.1
for _ in range(100): # Run both epoch=15 & epoch=100 
    weights, bias = update(x_train, y_train, weights, bias, learning_rate=0.1)
    # break


#%% (5) Testing

activation(x_test, weights, bias)
# y_test = np.where(y_test==0, "Iris-setosa", "Iris-versicolor")

print("Epochs = {}".format(_))
print('weights = ', weights, 'bias = ', bias)
print("y_test = {}".format(y_test))

