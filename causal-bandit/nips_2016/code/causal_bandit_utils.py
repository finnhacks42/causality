# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:18:04 2016

@author: finn
"""
import numpy as np
import datetime as dt

def now_string():
    return dt.now().strftime('%Y%m%d_%H%M')

def random_eta(n):
    eta = np.random.random(n)
    eta = eta/eta.sum()
    return eta
    
         
def argmax_rand(x):
    """ return the index of the maximum element in the array, ignoring nans. 
    If there are multiple max valued elements return 1 at random"""
    max_val = np.nanmax(x)
    indicies = np.where(x == max_val)[0]
    return np.random.choice(indicies) 

 

    
def prod_all_but_j2(vector):
    indx = np.range(len(vector))
    joint = np.prod(vector)
    joint_on_v = np.true_divide(joint,vector)
    for j in np.where(np.isnan(joint_on_v))[0]:
        joint_on_v[j] = np.prod(vector[indx != j])
    
    return joint_on_v
    

    

    