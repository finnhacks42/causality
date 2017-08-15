# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:15:43 2016

@author: finn
"""

import numpy as np
from math import sqrt, log
from enviroments import Parallel


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
            
        self.infrequent = self.estimate_infrequent(h)
        n = int(float(h)/len(self.infrequent))
        self.trials[self.infrequent] = n # note could be improved by adding to rather than reseting observation results - does not change worst case. 
        self.success[self.infrequent] = model.sample_multiple(self.infrequent,n)
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
        
class DiscreteContextualBandit(object):
    """ run a seperate bandit algorithm for each possible value of the context"""
    def __init__(self,num_contexts,bandit_cls):
        self.bandits = [bandit_cls() for i in range(num_contexts)] # create one for each
        
        
    def expected_regret(self,model):
        return model.best_reward()[self.contexts] - model.expected_reward_sequence(self.arms,self.contexts)
            
    def run(self,T,model):
        self.contexts = np.full(T,-1,dtype=int)
        self.arms = np.full(T,-1,dtype=int)
        for bandit in self.bandits:
            bandit.initialize(T,model)
        
        for t in xrange(T):
            context = model.sample_context()
            self.contexts[t] = context
            bandit = self.bandits[context]
            arm = bandit.select_action()
            reward = model.sample_conditional_reward(context,arm)
            bandit.update_reward(arm,reward)
        
        for context, bandit in enumerate(self.bandits):
            self.arms[self.contexts == context] = bandit.arms[0:bandit.t]
        
        
class ThompsonSampling(object):
    """ Sample actions via the Thomson sampling approach and return the empirically best arm 
        when the number of rounds is exhausted """
    label = "Thompson Sampling"
    
    def expected_reward(self,model):
        return model.expected_rewards[self.arms]
    
    def expected_regret(self,model):
        """ returns the expected regret of the actual sequence of arm choices as a vector of length T"""
        return max(model.expected_rewards) - self.expected_reward(model)
    
    def regret(self,model):
        return max(model.expected_rewards)*len(self.rewards) - self.rewards.sum()
        
    def initialize(self,T,model):
        self.trials = np.full(model.K,2,dtype=int)
        self.success = np.full(model.K,1,dtype=int)
        self.rewards = np.zeros(T)
        self.arms = np.full(T,-1,dtype=int)
        self.t = 0
        
    def select_action(self):
        fails = self.trials - self.success
        theta = np.random.beta(self.success,fails)
        arm = argmax_rand(theta)
        self.trials[arm] +=1 # keeps track of the number of times each arm played
        self.arms[self.t] = arm # keeps track of when each arm was played
        return arm
        
    def update_reward(self,arm,reward):
        self.success[arm]+= reward
        self.rewards[self.t] = reward
        self.t +=1
            
    def run(self,T,model):
        """
        Run Thompson sampling on the specified model for T timesteps.
        model: a model from which to sample rewards. 
               Should implement sample_multiple method and have expected_rewards property.
        T: the number of timesteps over which the algorithm should be run. 
        
        returns: the simple regret: the difference between the expected reward 
                 of the best arm and the expected reward of the arm the algorithm believes to be optimal
        """
        self.initialize(T,model) 
        
        for t in xrange(T):
            arm = self.select_action()
            reward = model.sample_multiple(arm,1)
            self.update_reward(arm,reward)
                
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
            self.rewards = None
            return np.nan
            
        self.rewards = np.zeros(T)
        self.arms = np.zeros(T)
        
        actions = range(0,model.K)
        self.trials = np.ones(model.K)
        self.success = model.sample_multiple(actions,1)
        self.arms[0:model.K] = np.asarray(actions)
        self.rewards[0:model.K] = self.success
        
        best_reward = max(model.expected_rewards)
       
        for t in range(model.K,T):
            arm = argmax_rand(self.upper_bound(t))
            self.trials[arm] += 1
            reward = model.sample_multiple(arm,1)
            self.success[arm] += reward
            self.arms[t] = arm
            self.rewards[t] = reward
            #self.regret[t] = best_reward - model.expected_rewards[arm]
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(mu)
        return best_reward - model.expected_rewards[self.best_action]
    
    
        
    
        
        
        

class AlphaUCB(UCB):
    """ 
    Implementation based on ... 
    UCB-1 corresponds to setting alpha = 4
    """
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


# Some models useful for sanity checks
# -------------------------------------------------------------------------------------------------

class ObservationalEstimate(object):
    """ Just observes for all actions, and then selects the arm with the best emprical mean. 
        Assumes P(Y|do(X)) = P(Y|X) as for ParallelCausal. Some actions may be entirely unexplored. """
    label = "Observational"
    
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        for t in xrange(T):
            x,y = model.sample(model.K-1)
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
        self.u = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
class UniformSampling(object):
    label = "Uniform"
    
    def run(self,T,model):
        trials_per_action = T/model.K
        success = model.sample_multiple(range(model.K),trials_per_action)
        self.u = np.true_divide(success,trials_per_action)
        self.best_action = argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
class RandomArm(object):
    label = "Random arm"
    
    def run(self,T,model):
        self.best_action = np.random.randint(0,model.K)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action] 
