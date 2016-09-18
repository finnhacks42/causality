# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:15:43 2016

@author: finn
"""

import numpy as np
from math import sqrt, log
from causal_bandit_utils import random_eta, argmax_rand
from scipy.optimize import minimize


class GeneralCausal(object):
    
    def run(self,T,model,eta,m):
        n = len(eta)
        self.B = sqrt(m*T/log(2.0*T*n))
        actions = range(n)
        print "possible actions",actions
        u = np.zeros(n)
        for t in xrange(T):
            a = np.random.choice(actions,p=eta)
            x,y = model.sample(a) #x is an array containing values for each variable
            pa = model.P(x)
            r = model.R(pa,eta)
            z = r*y #(r <= self.B)*r*y #Truncation turned off to run parallel bandit experiments
            u += z
        self.u = u/float(T)
        best_action = np.argmax(u)
        return model.optimal - model.expected_rewards[best_action]   
    
        
    def find_eta(self,model,length_eta = None):
        if length_eta is None:
            length_eta = model.K
        print "in here"
        eta0 = random_eta(length_eta)
        res = minimize(model.m, eta0,bounds = [(0.0,1.0)]*length_eta, constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0}),options={'disp': True},method='SLSQP')
        assert res.success, " optimization failed to converge"+res.message
        return res.x,res.fun

#TODO modify so this can be run on non parallel model    
class ParallelCausal(object):
    
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        h = T/2
        for t in range(h):
            x,y = model.sample(model.K) # do nothing
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
            
        infrequent = self.estimate_infrequent(h)
        n = int(float(h)/len(infrequent))
        self.trials[infrequent] = n # note could be improved by adding to rather than reseting observation results - does not change worst case. 
        self.success[infrequent] = model.sample_multiple(infrequent,n)
        u = np.true_divide(self.success,self.trials)
        best_action = argmax_rand(u)
        return model.optimal - model.expected_rewards[best_action]
   
            
    def estimate_infrequent(self,h):
        qij_hat = np.true_divide(self.trials,h)
        s_indx = np.argsort(qij_hat) #indexes of elements from s in sorted(s)
        m_hat = ParallelCausal.calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent
        

class LilUCB(object):
    """ Implementation based on """
    def run(self,T,model):
        return
       
 
class SuccessiveRejects(object):
    """ Implementation based on the paper 'Best Arm Identification in Multi-Armed Bandits',Audibert,Bubeck & Munos"""
    label = "Successive Reject"
    
    def run(self,T,model):
        
        if T <= model.K:
            return np.nan
        else:
            self.trials = np.zeros(model.K)
            self.success = np.zeros(model.K)
            self.actions = range(0,model.K)
            allocations = self.allocate(T,model.K)
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