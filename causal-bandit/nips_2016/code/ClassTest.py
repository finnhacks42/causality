# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:34:20 2016

@author: finn
"""

import numpy as np
class Animal(object):
    label = "Animal"
    def init(self):
        print self.noise()
        
    def noise(self):
        return "hmm"
        
class Dog(Animal):
    label = "Dog"
    def noise(self):
        return "woof"
        
def variance(data):
    m = data[0]
    s = 0
    n = data.shape[0]
    for k in range(1,n+1):
        x = data[k-1]
        m_next = m+(x - m)/k
        s_next = s+(x - m)*(x - m_next)
        m = m_next
        s = s_next
    return s/n
        
    
data = np.random.random(size = (100,3))

v = data.std(axis=0)**2
v2 = variance(data)