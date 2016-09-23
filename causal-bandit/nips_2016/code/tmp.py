# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:08:38 2016

@author: finn
"""
import numpy as np
s = 3
q = .4
sims = 1000000

results = np.zeros((sims,2))
for i in xrange(sims):
    z = np.random.binomial(1,np.full(s,q))
    v1 = z.mean()
    v2 = z.prod()
    results[i,0] = v1
    results[i,1] = v2

v2is0 = np.where(results[:,1]==0)[0]

m = results[v2is0].mean(axis=0)

p2 = q**(s)


print m[0], q/(1-p2) - p2, (q-p2)/(1-p2) 


