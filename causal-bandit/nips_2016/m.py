# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:42:54 2016

@author: finn
"""

import numpy as np

def m1(q):
    s = np.sort(np.minimum(q,1-q))
    for indx,value in enumerate(s[1:]): # first element can be anything
        th =1.0/(indx+1) # if indx=0 th=.5
        if value >= th:
            return indx+1
    return len(s)
    
def m2(q):
    s = np.minimum(q,1-q)
    for t in range(2,len(q)+1):
        I_t = (s < 1.0/(t)).sum()
        if I_t <= t:
            return t

    
qs = [np.zeros(10),np.ones(10),np.full(10,.5),np.full(10,.1),np.asarray([0,0,0,0,0,0,0,0,0,.2])]
for q in qs:
    print(m1(q),m2(q))

for i in range(10000):
    q = np.random.uniform(size=20)
    if m1(q) != m2(q):
        print("Difference",q)

