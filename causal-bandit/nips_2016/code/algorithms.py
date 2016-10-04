# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:15:43 2016

@author: finn
"""

import numpy as np
from math import sqrt, log
from scipy.stats import beta
from models import Parallel,ParallelConfoundedNoZAction,ScaleableParallelConfounded


def argmax_rand(x):
    """ return the index of the maximum element in the array, ignoring nans. 
    If there are multiple max valued elements return 1 at random"""
    max_val = np.nanmax(x)
    indicies = np.where(x == max_val)[0]
    return np.random.choice(indicies) 



class GeneralCausal(object):
    label = "Algorithm 2"
    
    def __init__(self,truncate = "clip"):
        self.truncate = truncate
        #self.label = "Algorithm 2-"+truncate

    def run(self,T,model):
        eta = model.eta
        m = model.m
        n = len(eta)
        self.B = sqrt(m*T/log(2.0*T*n))
        
        actions = range(n)
        u = np.zeros(n)
        for t in xrange(T):
            a = np.random.choice(actions,p=eta)
            x,y = model.sample(a) #x is an array containing values for each variable
            #y = y - model.get_costs()[a]
            pa = model.P(x)
            r = model.R(pa,eta)
            if self.truncate == "zero":
                z = (r<=self.B)*r*y
            elif self.truncate == "clip":
                z = np.minimum(r,self.B)*y
            else:
                z = r*y
                
            u += z
        self.u = u/float(T)
        r = self.u - model.get_costs()
        self.best_action = np.argmax(r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]   
    
        
class RandomArm(object):
    label = "Random arm"
    
    def run(self,T,model):
        self.best_action = np.random.randint(0,model.K)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action] 
  
class ParallelCausal(object):
    label = "Algorithm 1"
  
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        h = T/2
        for t in range(h):
            x,y = model.sample(model.K-1) # do nothing
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
            
        infrequent = self.estimate_infrequent(h)
        n = int(float(h)/len(infrequent))
        self.trials[infrequent] = n # note could be improved by adding to rather than reseting observation results - does not change worst case. 
        self.success[infrequent] = model.sample_multiple(infrequent,n)
        self.u = np.true_divide(self.success,self.trials)
        self.r = self.u - model.get_costs()
        self.best_action = argmax_rand(self.r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
   
            
    def estimate_infrequent(self,h):
        qij_hat = np.true_divide(self.trials,h)
        s_indx = np.argsort(qij_hat) #indexes of elements from s in sorted(s)
        m_hat = Parallel.calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent
        
 
class ThompsonSampling(object):
    """ Sample actions via the Thomson sampling approach and return the empirically best arm 
        when the number of rounds is exhausted """
    label = "Thompson Sampling"
    
    def run(self,T,model):
        self.trials = np.full(model.K,2,dtype=int)
        self.success = np.full(model.K,1,dtype=int)
        
        for t in xrange(T):
            fails = self.trials - self.success
            theta = np.random.beta(self.success,fails)
            arm = argmax_rand(theta)
            self.trials[arm] +=1
            self.success[arm]+= model.sample_multiple(arm,1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
        
        
class UCB(object):
    """ 
    Implements Generic UCB algorithm.
    """
    def run(self,T,model):
        if T <= model.K: # result is not defined if the horizon is shorter than the number of actions
            self.best_action = None
            return np.nan
        
        actions = range(0,model.K)
        self.trials = np.ones(model.K)
        self.success = model.sample_multiple(actions,1)
        
        for t in range(model.K,T):
            arm = argmax_rand(self.upper_bound(t))
            self.trials[arm] += 1
            self.success[arm] +=model.sample_multiple(arm,1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        

class LilUCB(UCB):
    """ 
    Implementation based on lil’UCB : An Optimal Exploration Algorithm for Multi-Armed Bandits
    Jamieson et al, COLT 2014.
    
    To convert the algorithm from fixed confidence to fixed budget, we use the halving trick.
    """
    label = "LilUCB"
    def __init__(self,epsilon,gamma,beta,delta,sigma = .5):
        
        self.e = epsilon
        self.g = gamma
        self.b = beta
        self.d = delta
        self.c1 = (1+beta)*(1+sqrt(epsilon))
        self.c2 = 2*(sigma**2)*(1+epsilon)
        
    @classmethod
    def theoretical(cls):
        """ Returns LilUCB with parameters supported by theory (from Jameison et al) """
        
        return cls(epsilon = .01,gamma = 9.0,beta = 1.0)
    
    @classmethod
    def heuristic(cls):
        """ Returns LilUCB with parameters heuristically shown to perform well (from Jamieson et al)"""
        
    def upper_bound(self,t):
        mu = np.true_divide(self.success,self.trials)
        ratio = np.true_divide(self.c2*np.log(np.log((1+self.e)*self.trials)/self.delta),self.trials)
        interval = self.c1*np.sqrt(ratio)
        return mu+interval
        

class AlphaUCB(UCB):
    """ Implementation based on ... """
    label = "UCB"
    
    def __init__(self,alpha):
        self.alpha = alpha
    
    def upper_bound(self,t):
        mu = np.true_divide(self.success,self.trials)
        interval = np.sqrt(self.alpha*np.log(t)/(2.0*self.trials))
        return mu+interval

        
class SuccessiveRejects(object):
    """ Implementation based on the paper 'Best Arm Identification in Multi-Armed Bandits',Audibert,Bubeck & Munos"""
    label = "Successive Reject"
    
    def run(self,T,model):
        
        if T <= model.K:
            self.best_action = None
            return np.nan
        else:
            self.trials = np.zeros(model.K)
            self.success = np.zeros(model.K)
            self.actions = range(0,model.K)
            self.allocations = self.allocate(T,model.K)
            self.rejected = np.zeros((model.K),dtype=bool)
            for k in range(0,model.K-1):
                nk = self.allocations[k]
                self.success[self.actions] += model.sample_multiple(self.actions,nk)
                self.trials[self.actions] += nk
                self.reject()
            
            assert len(self.actions) == 1, "number of arms remaining is: {0}, not 1.".format(len(self.actions))
            assert sum(self.trials) <= T,"number of pulls = {0}, exceeds T = {1}".format(sum(self.trials),T)
            self.best_action = self.actions[0]
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
    
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
        mu[self.rejected] = 2 # we don't want to reject the worst again
        min_val = np.min(mu)
        indicies = np.where(mu == min_val)[0] # these are the arms reported as worst
        return np.random.choice(indicies) # select one at random

if __name__ == "__main__":  
    
    
    
    N =5
    N1 = 1
    pz = .1
    pz = .2
    q = (.1,.9,.2,.8)
    pY = np.asarray([[.4,.4],[.7,.7]])
    epsilon=.1
    model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)
    alg = ThompsonSampling()
    alg2 = AlphaUCB(2)
    import time
    start = time.time()
    alg.run(1000,model)
    end = time.time()
    print end - start
        
    
    
    #model = ParallelConfoundedNoZAction.create(N,N1,pz,q,epsilon)
    #model.make_ith_arm_epsilon_best(epsilon,0)
    
    #alg.run(200,model)
  
    
#    sims = 1000
#    pulls = np.zeros(model.K,dtype=int)
#    regret = np.zeros(sims)
#    for s in range(1000):
#        regret[s] = alg.run(200,model)
#        pulls[alg.best_action] +=1
#    print regret.mean()