# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from math import sqrt,log,ceil
from numpy.random import binomial
import matplotlib
import matplotlib.pyplot as plt
from time import time
from datetime import datetime as dt
import pickle

# Prevents the use of Type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


REGRET_LABEL = "Regret"
HORIZON_LABEL = "T"
M_LABEL = "m(q)"

def now_string():
    return dt.now().strftime('%Y%m%d_%H%M')

def random_eta(n):
    eta = np.random.random(n)
    eta = eta/eta.sum()
    return eta
    
def most_unbalanced_q(N,m):
    q = np.full(N,1.0/m,dtype=float)
    q[0:m] = 0
    return q
    
def part_balanced_q(N,m):
    """ all but m of the variables have probability .5"""
    q = np.full(N,.5,dtype=float)
    q[0:m] = 0
    return q
        
def argmax_rand(x):
    """ return the index of the maximum element in the array, ignoring nans. 
    If there are multiple max valued elements return 1 at random"""
    max_val = np.nanmax(x)
    indicies = np.where(x == max_val)[0]
    return np.random.choice(indicies) 

def calculate_m(qij_sorted):
    for indx,value in enumerate(qij_sorted):
        if value >= 1.0/(indx+1):
            return indx
    return len(qij_sorted)/2      
    

class Parallel(object):
    def __init__(self,q,epsilon):
        """ actions are do(x_1 = 0)...do(x_N = 0),do(x_1=1)...do(N_1=1), do() """
        assert q[0] <= .5, "P(x_1 = 1) should be <= .5 to ensure worst case reward distribution can be created"
        self.q = q
        self.N = len(q) # number of variables
        self.K = 2*self.N+1 #number of actions
        self.pX = np.vstack((1.0-q,q))
        self.set_epsilon(epsilon)
        self.parent_vals = None

        
    def set_epsilon(self,epsilon):
        assert epsilon <= .5 ,"epsilon cannot exceed .5"
        self.epsilon = epsilon
        self.epsilon_minus = self.epsilon*self.q[0]/(1.0-self.q[0]) 
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
        result = np.hstack((pij,pobserve))
        return result
        
    def R(self,x,eta):
        pa = self.P(x)
        Q = (eta*pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
        
    def V(self,eta):
        if not self.parent_vals:
            self.parent_vals = map(np.asarray,product([0,1],repeat = self.N)) # all possible assignments to our N variables that are parents of Y
        va = np.zeros(self.K)  
        for x in self.parent_vals:
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
        mq = calculate_m(ordered)
        unbalanced = sort_order[0:mq]
        eta[unbalanced] = 1.0/(2*mq)
        return eta,mq
    

    
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
        m_hat = calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent
       

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
            z = r*y #(r <= self.B)*r*y #Truncation turned off to run parallel bandit experiments
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


def setup_regret_vs_T(N,epsilon=None,Tstep=None,TperK=10):
    a = 4.0
    q = part_balanced_q(N,2) 
    if epsilon:
        model = Parallel(q,epsilon)
        Tmin = 10
        vary_epsilon = False
    else:
        model =  Parallel(q,.5)
        Tmin = int(ceil(4*model.K/a))
        vary_epsilon = True
    
    Tstep = model.K if not Tstep else Tstep
    eta,mq = model.analytic_eta()
     
    T_vals = range(Tmin,TperK*model.K,Tstep)
    return model,eta,mq,T_vals,vary_epsilon,a
    
    
def regret_vs_T(N,simulations,epsilon=None,Tstep=None,TperK=10):
    model,eta,mq,T_vals,vary_epsilon,a =setup_regret_vs_T(N,epsilon=epsilon,Tstep=Tstep,TperK=TperK)
    
    causal = GeneralCausal()
    causal_parallel = ParallelCausal()
    baseline  = SuccessiveRejects()
    
    ts = time()   
    regret = np.zeros((len(T_vals),4,simulations))
    for s in xrange(simulations):
        if s % 100 == 0:
                print s
        for T_indx,T in enumerate(T_vals): 
            if vary_epsilon: #variable epsilon
                epsilon = sqrt(model.K/(a*T))
                model.set_epsilon(epsilon)
            regret[T_indx,0,s] = causal.run(T,model,eta,mq)
            regret[T_indx,1,s] = causal_parallel.run(T,model)
            regret[T_indx,2,s] = baseline.run(T,model)
            regret[T_indx,3,s] = epsilon
            
    te = time()
    print 'took: %2.4f sec' % (te-ts)
    return regret,T_vals,N,a  
    
def plot_regret(regret,xvals):
    simulations = regret.shape[-1]
    mean = regret.mean(axis=2)       
    error = 3*regret.std(axis=2)/sqrt(simulations)    
    fig,ax = plt.subplots(figsize=(4.5,3))
    ax.errorbar(xvals,mean[:,1],yerr=error[:,1], label="Algorithm 1",linestyle="",marker="o",markersize=6)  
    ax.errorbar(xvals,mean[:,0],yerr=error[:,0], label="Algorithm 2",linestyle="",marker="s",markersize=5) 
    ax.errorbar(xvals,mean[:,2],yerr=error[:,2], label="Successive Rejects",linestyle="",marker="D",markersize=5) 
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    return fig,ax


def plot_regret_versus_T(regret,T_vals,N,a,experiment_number):  
    fig,ax = plot_regret(regret,T_vals) 
    ax.set_xlabel(HORIZON_LABEL)  
    ax.set_ylim(bottom=0)
   
    if experiment_number == 2:
        ax.set_xticks(np.arange(0,1001,200))  
        ax.set_yticks(np.arange(0,.41,.1))
        fig_name = "experiment2_N{0}_a{1}_s{2}_{3}.pdf".format(N,int(a),regret.shape[-1],now_string())
    else:
        fig_name = "experiment3_N{0}_s{1}_{2}.pdf".format(N,regret.shape[-1],now_string())
        ax.set_yticks(np.arange(0,.31,.1))
    
    #ax.legend(loc="upper right",numpoints=1)
    fig.savefig(fig_name, bbox_inches='tight') 
     
    
def plot_regret_versus_m(regret,m_vals,T):
    fig,ax = plot_regret(regret,m_vals)
    ax.set_xlabel(M_LABEL)
    ax.set_yticks(np.arange(0,.31,.1))
    ax.legend(loc="lower right",numpoints=1)
    fig_name = "exp_regret_vs_m_N{0}_T{1}_s{2}_{3}.pdf".format(N,T,regret.shape[-1],now_string())
    fig.savefig(fig_name, bbox_inches='tight') 
        
   
def regret_vs_m(N,epsilon,simulations,T):
    m_vals = range(2,N,2)
    causal = GeneralCausal()
    causal_parallel = ParallelCausal()
    baseline  = SuccessiveRejects()
    
    ts = time()   
    regret = np.zeros((len(m_vals),3,simulations))
    for s in xrange(simulations):
        if s % 100 == 0:
                print s
        for m_indx,m in enumerate(m_vals): 
            q = part_balanced_q(N,m)
            model = Parallel(q,epsilon)
            eta,mq = model.analytic_eta()
            regret[m_indx,0,s] = causal.run(T,model,eta,mq)
            regret[m_indx,1,s] = causal_parallel.run(T,model)
            regret[m_indx,2,s] = baseline.run(T,model)
            
    te = time()
    print 'took: %2.4f sec' % (te-ts)
    return regret,m_vals,T
    




## Experiment 1
N = 50
epsilon = .3
#simulations = 1000
T = 400
#regret,mean,error = regret_vs_m(N,epsilon,simulations,T)
#with open("experiment1_{0}.pickle".format(now_string()),"wb") as out:
#    pickle.dump(regret,out)

with open('data/experiment1_20160518_1339.pickle','rb') as f:
    regret1 = pickle.load(f)
    m_vals = range(2,N,2)
    plot_regret_versus_m(regret1,m_vals,T)

#
#
## Experiment 2
N= 50
#simulations = 1000
#regret2,mean2,error2 = regret_vs_T(N,simulations,epsilon=None,Tstep=None,TperK=10)
#with open("experiment2_{0}.pickle".format(now_string()),"wb") as out:
#    pickle.dump(regret2,out)

with open('data/experiment2_20160518_1206.pickle',"rb") as f:
    regret2 = pickle.load(f)
    model,eta,mq,T_vals,vary_epsilon,a = setup_regret_vs_T(N,epsilon=None,Tstep=None,TperK=10)
    plot_regret_versus_T(regret2,T_vals,N,a,2)

#
### Experiment 3
N = 50
epsilon = .3
#simulations = 1000
#regret3,mean3,error3 = regret_vs_T(N,simulations,epsilon=epsilon,Tstep=25,TperK=6)
#with open("experiment3_{0}.pickle".format(now_string()),"wb") as out:
#    pickle.dump(regret3, out)

with open('data/experiment3_20160518_0117.pickle',"rb") as f:
    regret3 = pickle.load(f)
    model,eta,mq,T_vals,vary_epsilon,a = setup_regret_vs_T(N,epsilon=epsilon,Tstep=25,TperK=6)
    plot_regret_versus_T(regret3,T_vals,N,a,3)

    
    
    

