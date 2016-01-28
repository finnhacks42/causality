# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:11:53 2016

@author: finn
"""

from numpy.random import binomial
import numpy as np
import matplotlib.pyplot as plt


pA = .6 #Probability that A = 0
pB = np.array([.5,.8]) #Probability that B = 0 given A=0,A=1
pC = np.array([.5,.2])
pY = np.array([[.2,.9],[.6,.8]]) # P(Y|B,C)

pYis1givenDoBis0 = 1 - (pY[0][0]*pC[0]*pA+ pY[0][0]*pC[1]*(1-pA)+pY[0][1]*(1-pC[0])*pA+ pY[0][1]*(1-pC[1])*(1-pA))


T = 500
simulations = 1000
data = np.zeros((T,4),dtype=int) 
estimates = np.zeros((simulations,2))
for s in xrange(simulations):
    for t in range(T):
        a = 1 -binomial(1,pA)
        b = 1 -binomial(1,pB[a])
        c = 1 -binomial(1,pC[a])
        y = 1- binomial(1,pY[b][c])
        data[t] = [a,b,c,y]
        
    # now use data to build some estimators
    ais0 = data[:,0] == 0
    bis0 = data[:,1] == 0
    t1 = np.logical_and(bis0,ais0)
    t2 = np.logical_and(bis0,np.logical_not(ais0)) 
    u = pA*np.sum(data[:,-1]*t1)/float(np.sum(t1)) + (1-pA)* np.sum(data[:,-1]*t2)/float(np.sum(t2)) 
    
    zt =  np.true_divide(data[:,-1]*bis0,pB[data[:,0]])   
    u2 = (1.0/T)*np.sum(zt) 
    
    estimates[s,0] = u
    estimates[s,1] = u2



plt.subplot(2, 1, 1)
plt.hist(estimates[:,0],bins = 100, range = (0,1))
plt.axvline(x=pYis1givenDoBis0,color='red',linewidth=2)
plt.title('Estimators for P(Y|do(B=0))')
plt.ylabel('Backdoor path')
plt.subplot(2, 1, 2)
plt.hist(estimates[:,1],bins = 100, range = (0,1))
plt.axvline(x=pYis1givenDoBis0,color='red',linewidth=2)
plt.xlabel('$\hat{\mu}$')
plt.ylabel('Importance weighted')
plt.show()

print "True value", pYis1givenDoBis0
print np.mean(estimates,axis=0)
print np.std(estimates,axis=0)

