# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:31:59 2016

@author: finn
"""
import numpy as np

samples = 10


x_means = [1,2]
x_var = [.1,4]
num_features = len(x_means)

x = np.empty((samples,num_features+1))
for i in range(num_features):
    x[:,i] = np.random.normal(x_means[i],x_var[i],size=samples)
x[:,-1] = 1

w = [.1,5,2]

# how much each feature shifts the mean will be important for its selection via the balanced regression approach. (not for LASSO)

t = 







