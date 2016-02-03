# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
import sys

class BalancedParallel(object):
    def __init__(self,N,q):
        self.N = N
        self.pX = np.vstack((1-q,q))
       
    def P(self,x):
        """ calculate vector of P_a for each action a """
        indx = np.arange(len(x))
        ps = self.pX[x,indx] #probability of xi for each i
        joint = ps.prod()
        pi = np.true_divide(joint,ps) # will be nan for elements for which ps is 0 -(should never happen if sampling)
        for j in np.where(np.isnan(pi))[0]:
            pi[j] = np.prod(ps[indx != j])
        
        pij = np.vstack((pi,pi))
        pij[1-x,indx] = 0
        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
        pobserve = joint# the probability of x given do()
        return np.hstack((pij,pobserve))
        
    def V(self,eta):
        xvals = map(np.asarray,product([0,1],repeat = self.N)) # all possible assignments to our N variables that are parents of Y
        va = np.zeros(2*self.N + 1)  
        for x in xvals:
            pa = self.P(x)
            Q = (eta*pa).sum()
            ratio = np.true_divide(pa**2,Q)
            ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
            va += ratio
            
        return va 


BIG_FLOAT = sys.float_info.max   
        
N = 6 # number of parents of Y
n = 2*N+1 # number of actions
q = np.full(N,.5)
q[0:2] = 1
print q
model = BalancedParallel(N,q)

eta0 = np.full(n,1.0/n)
eta1 = np.zeros(n)
eta1[-1]=1 
eta2 = np.zeros(n)
eta2[0] = 1


print model.V(eta0)

def random_eta(n):
    eta = np.random.random(n)
    eta = eta/eta.sum()
    return eta

def maxV(eta):
    penalty = 5000*(eta.sum()-1)**2
    maxV = model.V(eta).max()
    if not np.isfinite(maxV):
        return BIG_FLOAT
    return maxV+penalty # consider penalizing eta going beyond bounds     


#constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0})
results = []
for t in range(10):
    eta0 = random_eta(n)
    res = minimize(maxV,eta0,bounds = [(0.0,1.0)]*n,method='L-BFGS-B',options={'disp': True,'ftol' : 10 * np.finfo(float).eps})
    results.append(res)
    print "setting",res.x,sum(res.x)
    print "V", model.V(res.x)
    print "Vmax",max(model.V(res.x))
 



#print sum(res.x)
# need do and doZ in there as actions too...
    
 
#    def V(self,eta):
#        xvals = map(np.asarray,product([0,1],repeat = self.N-1)) # all possible assignments to our N-1 variables that are parents of Y
#        va = np.zeros(2*self.N + 1)        
#        for x in xvals:
#            pa = self.P(x)
#            Q = (eta*pa).sum()
#            va += np.true_divide(pa**2,Q)
#        return va 

   
#    def P(self,x):
#        assert len(x) == self.pX.shape[1]
#        indices = np.arange(len(x))
#        ps = self.pX[x,indices] # probability of xi for each i
#        pi = np.asarray([np.prod(ps[indices != i]) for i in indices])
#        pij = np.vstack((pi,pi))
#        pij[1-x,indices] = 0 # these are the probability of observing the given x for each action do(Xi=j) 2*(N-1) array
#        
#        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
#        pobserve = np.prod(ps) # the probability of x given do()
#        pxz0 = np.prod(self.pxz[0,indices,x]) # the probabilities of x given do(z = 0)
#        pxz1 = np.prod(self.pxz[1,indices,x]) # the probabilities of x given do(z = 1)
#        pa = np.hstack((pij,pxz0,pxz1,pobserve))
#        
#        return pa
#    

