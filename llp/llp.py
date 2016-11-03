# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:06:18 2016

@author: finn
"""
import numpy as np
from simulator import Linear
from estimators import StandardEstimator, CausalLLMEstimator, EMEstimator




# create some simulated data

n = 100
d = 5
data_model = Linear(d)
x_treat,x_control,y_treat,y_control = data_model.sample(n)



llm = CausalLLMEstimator()
llm.fit(x_treat,x_control,y_treat,y_control)

logistic = StandardEstimator()
logistic.fit(x_treat,x_control,y_treat,y_control)

em = EMEstimator()
em.fit(x_treat,x_control,y_treat,y_control)
















