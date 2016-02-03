# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize
from math import sqrt
from math import log
from numpy.random import binomial
import matplotlib.pyplot as plt
from time import time



def random_eta(n):
    eta = np.random.random(n)
    eta = eta/eta.sum()
    return eta
    
def most_unbalanced_q(N,m):
    q = np.full(N,1.0/m,dtype=float)
    q[0:m] = 0
    return q
    

def maxV(eta):
    maxV = model.V(eta).max()
    if not np.isfinite(maxV):
        return np.inf
    return maxV#+penalty # consider penalizing eta going beyond bounds 

class Parallel(object):
    def __init__(self,q,epsilon):
        """ actions are do(x_1 = 0)...do(x_N = 0),do(x_1=1)...do(N_1=1), do() """
        assert q[0] <= .5, "P(x_1 = 1) should be <= .5 to ensure worst case reward distribution can be created"
        self.epsilon = epsilon
        self.epsilon_minus = self.epsilon*q[0]/(1.0-q[0]) 
        self.N = len(q) # number of variables
        self.K = 2*self.N+1 #number of actions
        self.pX = np.vstack((1.0-q,q))
        
        self.expected_rewards = np.full(self.K,.5)
        self.expected_rewards[0] = .5 - self.epsilon_minus
        self.expected_rewards[self.N] = .5+self.epsilon
        self.optimal = .5+self.epsilon
         
    def sample(self,action):
        x = binomial(1,self.pX[1,:])
        if action != self.K - 1: # everything except the do() action
            i,j = action % self.N, action/self.N
            x[i] = j
        y = binomial(1,self.pYgivenX(x))
        return x,y
    
    def sample_multiple(self,actions,n):
        """ sample the specified actions, n times each """
        return binomial(n,self.expected_rewards[actions])
        
    def pYgivenX(self,x):
        if x[0] == 1:
            return .5+self.epsilon
        return .5 -self.epsilon_minus
        
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
        
    def R(self,x,eta):
        pa = self.P(x)
        Q = (eta*pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
        
    def V(self,eta):
        xvals = map(np.asarray,product([0,1],repeat = self.N)) # all possible assignments to our N variables that are parents of Y
        va = np.zeros(self.K)  
        for x in xvals:
            pa = self.P(x)
            Q = (eta*pa).sum()
            ratio = np.true_divide(pa**2,Q)
            ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
            va += ratio         
        return va 
    
    def m(self,eta):
        maxV = self.V(eta).max()
        assert not np.isnan(maxV), "m should not be nan"
        return maxV
    
    def analytic_eta(self):
        eta = np.zeros(self.K)
        eta[-1] =.5
        probs = self.pX[:,:].reshape((self.N*2,)) # reshape such that first N are do(Xi=0)
        sort_order = np.argsort(probs)
        ordered = probs[sort_order]
        mq = self.N
        for indx,value in enumerate(ordered):
            if value >= 1.0/(indx+1):
                mq = indx
                break
        unbalanced = sort_order[0:mq]
        eta[unbalanced] = 1.0/(2*mq)
        return eta,mq
    

        
        
    
 

class GeneralCausal(object):
    
    def run(self,T,model,eta,m):
        n = len(eta)
        self.B = sqrt(m*T/log(2.0*T*n))
        actions = range(n)
        u = np.zeros(n)
        for t in xrange(T):
            a = np.random.choice(actions,p=eta)
            x,y = model.sample(a) #x is an array containing values for each variable
            r = model.R(x,eta)
            z = (r <= self.B)*r*y
            u += z
        self.u = u/float(T)
        best_action = np.argmax(u)
        return model.optimal - model.expected_rewards[best_action]
        
 
class SuccessiveRejects(object):
    """ Implementation based on the paper 'Best Arm Identification in Multi-Armed Bandits',Audibert,Bubeck & Munos"""
    label = "Successive Reject"
    
    def run(self,T,model):
        if T <= model.K:
            return np.nan
        allocations = self.allocate(T,model.K)
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        self.actions = range(0,model.K)
        self.rejected = np.zeros((model.K),dtype=bool)
        for k in range(0,model.K-1):
            nk = allocations[k]
            self.success[self.actions] += model.sample_multiple(self.actions,nk)
            self.trials[self.actions] += nk
            self.reject()
        assert len(self.actions == 1), "number of arms remaining is: {0}, not 1.".format(len(self.actions))
        assert sum(self.trials <= T),"number of pulls = {0}, exceeds T = {1}".format(sum(self.trials),T)
        best_action = self.actions[0]
        return model.optimal - model.expected_rewards[best_action]
    
    def allocate(self,T,K):
        logK = .5 + np.true_divide(1,range(2,K+1)).sum()
        n = np.zeros((K),dtype=int)
        n[1:] =  np.ceil((1.0/logK)*np.true_divide((T - K),range(K,1,-1)))
        allocations = np.diff(n)
        return allocations
                       
    def reject(self):
        worst_arm = self.worst()
        self.rejected[worst_arm] = True
        self.actions = np.where(~self.rejected)[0] 
        
    def worst(self):
        mu = np.true_divide(self.success,self.trials)
        mu[self.rejected] = 1 # we don't want to reject the worst again
        min_val = np.min(mu)
        indicies = np.where(mu == min_val)[0] # these are the arms reported as worst
        return np.random.choice(indicies) # select one at random



def find_eta(model):
    eta0 = random_eta(model.K)
    res = minimize(model.m, eta0,bounds = [(0.0,1.0)]*model.K, constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0}),options={'disp': True},method='SLSQP')
    assert res.success, " optimization failed to converge"+res.message
    return res.x,res.fun
                                
N = 30
q = most_unbalanced_q(N,2) #TODO fix most unbalanced (N,1)
epsilon = .1
model = Parallel(q,epsilon)
eta,mq = model.analytic_eta()

model2 = Parallel(np.full(N,.5),epsilon) #balanced q
eta2 = np.zeros(model2.K)
eta2[-1] = 1.0 # just observe strategy 

#print model.m(eta)
#print model2.m(eta2)



T_vals = range(10,500,10)
simulations = 5000
causal = GeneralCausal()
baseline  = SuccessiveRejects()

ts = time()   
regret = np.zeros((len(T_vals),3,simulations))
for s in xrange(simulations):
    if s % 100 == 0:
            print s
    for T_indx,T in enumerate(T_vals):     
        regret[T_indx,0,s] = causal.run(T,model,eta,4)
        regret[T_indx,1,s] = causal.run(T,model2,eta2,2)
        regret[T_indx,2,s] = baseline.run(T,model)
        
te = time()
print 'took: %2.4f sec' % (te-ts)
mean = regret.mean(axis=2)       
plt.plot(T_vals,mean)

error = 3*regret.std(axis=2)/sqrt(simulations)
fig,ax = plt.subplots()
ax.errorbar(T_vals,mean[:,0],yerr=error[:,0], label="Causal m(eta) = 4",linestyle="",marker="o")    
ax.errorbar(T_vals,mean[:,2],yerr=error[:,2], label="Successive Rejects",linestyle="",marker="D") 
ax.legend(loc="lower left",numpoints=1)
   

#
##constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0})
#results = []
##'ftol' : 10 * np.finfo(float).eps
#o = open("converge3.txt","w")
#for t in range(30):
#    eta0 = random_eta(n)
#    res = minimize(maxV,eta0,bounds = [(0.0,1.0)]*n, constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0}),options={'disp': True},method='SLSQP')#, method='L-BFGS-B',
#    #res = minimize(maxV,eta0, options={'disp': True})
#    results.append(res)
#    if res.success:
#        result =["%.4f"%float(x) for x in res.x]
#        print result
#        data = "["+",".join(result)+"],"+str(res.x.sum())+", "+str(res.fun)+"\n"
#        o.write(data)
#        o.flush()
#    else:
#        print "FAIL"
#    
#
#results.sort(key=lambda x:x.fun)
#for r in results:
#    print ["%.3f" % float(v) for v in r.jac]

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

