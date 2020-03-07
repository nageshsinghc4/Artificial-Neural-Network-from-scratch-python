#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : Artificial Neural Network with 1 input and 1 output layer from scratch
Created on Wed Oct  9 17:14:14 2019
@author: nageshsinghchauhan
"""

import numpy as np

#create input data
input_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1],[0,1,1],[0,1,0]]) #independent variables X
labels = np.array([[1,0,0,1,1,0,1]]) #dependent variable y
labels = labels.reshape(7,1)

#Define Hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

#Define Activation Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Define derivative of Sigmoid activation function
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

#Train your model
for epoch in range(25000):
    inputs = input_set
    XW = np.dot(inputs, weights)+ bias
    z = sigmoid(XW)
    error = z - labels
    print(error.sum())
    dcost_dpred = error
    dpred_dz = sigmoid_derivative(z)
    z_del = dcost_dpred * dpred_dz
    inputs = input_set.T
    weights = weights - lr*np.dot(inputs, z_del)
    
    for num in z_del:
        bias -= lr*num

#Prediction
single_pt = np.array([0,1,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)
"""

# NN with one hidden layer
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.10)
plt.figure(figsize=(7,4))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
