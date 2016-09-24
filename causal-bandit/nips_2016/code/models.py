# -*- coding: utf-8 -*-
"""
The classes in this file model the enviroment (data generating process) and store the true value of the reward distribution
over actions. The true reward distribution is not known to the algorithms but is required to calculate regret.

Models should extend the Model class and must implement:
Attributes
- expected_rewards: a numpy array containing the true expected reward for each action
- optimal: max(expected_rewards)
- K: the size of the action space
- parent_assignments: a list of numpy arrays where each array is a possible assignment of values to the parents of Y.

Methods
- P(x): list of length K, returns the probability of the given assignment ,x, to the parents of Y for given each action. 
- sample(action): samples from the conditional distribution over the parents of Y and Y given the specified action index.
  returns X (numpy array of length num_parents(Y)), Y (float)

@author: finn
"""
import numpy as np
from itertools import product
from numpy.random import binomial
from scipy.optimize import minimize
from scipy.special import expit


np.set_printoptions(precision=6,suppress=True,linewidth=200)
def prod_all_but_j(vector):
    """ returns a vector where the jth term is the product of all the entries except the jth one """
    zeros = np.where(vector==0)[0]
    if len(zeros) > 1:
        return np.zeros(len(vector))
    if len(zeros) == 1:
        result = np.zeros(len(vector))
        j = zeros[0]
        result[j] = np.prod(vector[np.arange(len(vector)) != j])
        return result

    joint = np.prod(vector)
    return np.true_divide(joint,vector)
    

class Model(object):
                 
    def _expected_Y(self):
        """ Calculate the expected value of Y (over x sampled from p(x|a)) for each action """
        return np.dot(self.PY,self.A)
        
    def set_action_costs(self,costs):
        """ 
        update expected rewards to to account for action costs.
        costs should be an array of length K specifying the cost for each action.
        The expcted reward is E[Y|a] - cost(a). 
        If no costs are specified they are assume zero for all actions.
        """
        self.costs = costs
        self.expected_rewards = self.expected_Y - costs
        self.optimal = max(self.expected_rewards)
        
    def make_first_arm_epsilon_best(self,epsilon):
        """ adjusts the costs such that all arms have expected reward .5, expect the first one which has reward .5 + epsilon """
        costs = self.expected_Y - 0.5
        costs[0] -= epsilon
        self.set_action_costs(costs)
        
    
    def pre_compute(self,compute_py = True):
        """ 
        pre-computes expensive results 
        A is an lxk matrix such that A[i,j] = P(ith assignment | jth action)
        PY is an lx1 vector such that PY[i] = P(Y|ith assignment)
        """

        self.get_parent_assignments()
 
        A = np.zeros((len(self.parent_assignments),self.K))
        if compute_py:
            self.PY = np.zeros(len(self.parent_assignments))
        
        for indx,x in enumerate(self.parent_assignments):
            A[indx,:] = self.P(x)
            if compute_py:
                self.PY[indx] = self.pYgivenX(x)
            
        self.A = A
        self.A2T = (self.A**2).T
        
        self.expected_Y = self._expected_Y()
        self.expected_rewards = self.expected_Y
        self.optimal = max(self.expected_rewards)
        self.eta,self.m = self.find_eta()
        self.eta = self.eta/self.eta.sum() # random choice demands more accuracy than contraint in minimizer
        
    def clean_up(self):
        self.A = None
        self.A2T = None
        self.PY = None
        
    def get_costs(self):
        if not hasattr(self,"costs"):
            self.costs = np.zeros(self.K)
        return self.costs
        
    def get_parent_assignments(self):
        if not hasattr(self,"parent_assignments") or self.parent_assignments is None:
            self.parent_assignments = Model.generate_binary_assignments(self.N)
    
        return self.parent_assignments
    
    @classmethod
    def generate_binary_assignments(cls,N):
        """ generate all possible binary assignments to the N parents of Y. """
        return map(np.asarray,product([0,1],repeat = N))
        
    
    def R(self,pa,eta):
        """ returns the ratio of the probability of the given assignment under each action to the probability under the eta weighted sum of actions. """
        Q = (eta*pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
               
        
    def V(self,eta):
        """ returns a vector of length K with the expected value of R (over x sampled from p(x|a)) for each action a """
        #with np.errstate(divide='ignore'):        
        u = np.true_divide(1.0,np.dot(self.A,eta))
        u = np.nan_to_num(u) # converts infinities to very large numbers such that multiplying by 0 gives 0
        v = np.dot(self.A2T,u)
        return v
    
    def m_eta(self,eta):
        """ The maximum value of V"""
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m should not be nan, \n{0}\n{1}".format(eta,V)
        return maxV
        
    def random_eta(self):
        eta = np.random.random(self.K)
        return eta/eta.sum()
        
    def _minimize(self,tol = 1e-6):
        eta0 = self.random_eta()
        constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0})
        #options={'disp': True}
        res = minimize(self.m_eta, eta0,bounds = [(0.0,1.0)]*self.K, constraints = constraints, method='SLSQP')
        return res
        
    def find_eta(self,tol = 1e-10,min_starts = 3, max_starts = 10):
        m = self.K + 1
        eta = None
        starts = 0
        success = 0
        while success < min_starts and starts < max_starts:
            res = self._minimize(tol)            
            if res.success and res.fun <= self.K:
                success +=1
                if res.fun < m:
                    m = res.fun
                    eta = res.x
            starts +=1
        
        if eta is None:
            raise Exception("optimisation failed")
    
        return eta,m
             
    def sample_multiple(self,actions,n):
        """ draws n samples from the reward distributions of the specified actions. """
        return binomial(n,self.expected_rewards[actions])
        


class Parallel(Model):
    """ Parallel model as described in the paper """
    def __init__(self,q,epsilon):
        """ actions are do(x_1 = 0)...do(x_N = 0),do(x_1=1)...do(N_1=1), do() """
        assert q[0] <= .5, "P(x_1 = 1) should be <= .5 to ensure worst case reward distribution can be created"
        self.q = q
        self.N = len(q) # number of X variables (parents of Y)
        self.K = 2*self.N+1 #number of actions
        self.pX = np.vstack((1.0-q,q))
        self.set_epsilon(epsilon)
        self.eta,self.m = self.analytic_eta()
    
    @classmethod
    def create(cls,N,m,epsilon):
        q = cls.part_balanced_q(N,m)
        return cls(q,epsilon)
        
    @classmethod
    def most_unbalanced_q(cls,N,m):
        q = np.full(N,1.0/m,dtype=float)
        q[0:m] = 0
        return q
    
    @classmethod
    def part_balanced_q(cls,N,m):
        """ all but m of the variables have probability .5"""
        q = np.full(N,.5,dtype=float)
        q[0:m] = 0
        return q
    
    @staticmethod
    def calculate_m(qij_sorted):
        for indx,value in enumerate(qij_sorted):
            if value >= 1.0/(indx+1):
                return indx
        return len(qij_sorted)/2 
        
    
        
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
        y = binomial(1,self.pYgivenX(x))-self.get_costs()[action]
        return x,y
    
        
    def pYgivenX(self,x):
        if x[0] == 1:
            return .5+self.epsilon
        return .5 -self.epsilon_minus
        
    def P(self,x):
        """ calculate vector of P_a for each action a """
        indx = np.arange(len(x))
        ps = self.pX[x,indx] #probability of P(X_i = x_i) for each i given do()
        joint = ps.prod() # probability of x given do()
        pi = np.true_divide(joint,ps) # will be nan for elements for which ps is 0 
        for j in np.where(np.isnan(pi))[0]:
            pi[j] = np.prod(ps[indx != j]) 
        pij = np.vstack((pi,pi))
        pij[1-x,indx] = 0 # now this is the probability of x given do(x_i=j)
        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
        result = np.hstack((pij,joint))
        return result
        
    def analytic_eta(self):
        eta = np.zeros(self.K)
        eta[-1] =.5
        probs = self.pX[:,:].reshape((self.N*2,)) # reshape such that first N are do(Xi=0)
        sort_order = np.argsort(probs)
        ordered = probs[sort_order]
        mq = Parallel.calculate_m(ordered)
        unbalanced = sort_order[0:mq]
        eta[unbalanced] = 1.0/(2*mq)
        return eta,mq

      
           
class ParallelConfounded(Model):
    """ Represents a parallel bandit with one common confounder. Z ->(X1 ... XN) and (X1,...,XN) -> Y 
        Actions are do(x_1 = 0),...,do(x_N = 0), do(x_1=1),...,do(x_N = 1),do(Z=0),do(Z=1),do()"""
    
    def __init__(self,q10,q11,q20,q21,pZ,N1,N2,epsilon):
        self._init_pre_action(q10,q11,q20,q21,pZ,N1,N2,epsilon)
        self.K = 2*self.N + 3        
        self.pre_compute()  
        
    def _init_pre_action(self,q10,q11,q20,q21,pZ,N1,N2,epsilon):
        """ The initialization that should occur regardless of whether we can act on Z """
        pXgivenZ0 = np.hstack((np.full(N1,q10),np.full(N2,q20)))
        pXgivenZ1 = np.hstack((np.full(N1,q11),np.full(N2,q21)))
        self.pX0 = np.vstack((1.0-pXgivenZ0,pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        self.pX1 = np.vstack((1.0-pXgivenZ1,pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        self.pXgivenZ = np.stack((self.pX0,self.pX1),axis=2) # PXgivenZ[i,j,k] = P(X_i=j|Z=k)
    
        self.N1 = N1
        self.N2 = N2
        self.q10,self.q11,self.q20,self.q21 = q10,q11,q20,q21
        self.N = len(pXgivenZ0)
        self.indx = np.arange(self.N)
        self.pZ = pZ
     
        y_weights = np.full(self.N,1.0/self.N)
        y_weights[0] = -1  
        y_weights[self.N-1] = 1
        self.y_weights = y_weights
        
    
    def __str__(self):
        string = "ParallelConfounded_mis{0:.1f}_Nis{1}_N1is{2}_qis{3:.1f}_{4:.1f}_{5:.1f}_{6:.1f}_pzis{7:.1f}_epsilonis{8:.1f}".format(self.m,self.N,self.N1,self.q10,self.q11,self.q20,self.q21,self.pZ,self.epsilon)
        return string.replace(".","-")
        
               
    @classmethod
    def create(cls,N,N1,pz,q,epsilon):
        """ builds ParallelConfounded model"""
        q10,q11,q20,q21 = q
        N2 = N - N1
        model = cls(q10,q11,q20,q21,pz,N1,N2,epsilon)
        model.make_first_arm_epsilon_best(epsilon)
        return model
        
        
    def pYgivenX(self,x):
        s = np.dot(x,self.y_weights)
        return expit(s)

        
    def sample(self,action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        if action == 2*self.N+1: # do(z = 1)
            z = 1       
        elif action == 2*self.N: # do(z = 0)
            z = 0     
        else: # we are not setting z
            z = binomial(1,self.pZ)
        
        x = binomial(1,self.pXgivenZ[1,:,z]) # PXgivenZ[j,i,k] = P(X_i=j|Z=k)
        
        if action < 2*self.N: # setting x_i = j
             i,j = action % self.N, action/self.N
             x[i] = j
             
        y = binomial(1,self.pYgivenX(x)) - self.get_costs()[action]
        
        return x,y
        
              
    def P(self,x):
        """ calculate P(X = x|a) for each action a. 
            x is an array of length N specifiying an assignment to the parents of Y
            returns a vector of length K. 
        """
        pz1 = self.pXgivenZ[x,self.indx,1]
        pz0 = self.pXgivenZ[x,self.indx,0]
    
        p_obs = self.pZ*pz1.prod()+(1-self.pZ)*pz0.prod()
        
        # for do(x_i = j)
        joint_z0 = prod_all_but_j(pz0) # vector of length N
        joint_z1 = prod_all_but_j(pz1) 
        p = self.pZ * joint_z1+ (1-self.pZ) * joint_z0  
        pij = np.vstack((p,p))
        pij[1-x,self.indx] = 0 # 2*N array, pij[i,j] = P(X=x|do(X_i=j)) = d(X_i-j)*prod_k!=j(X_k = x_k)
        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
        
        result = np.hstack((pij,pz0.prod(),pz1.prod(),p_obs))
        return result
        
 
    def random_eta_short(self):
        weights = self.weights()
        eta0 = np.random.random(len(weights))
        eta0 = eta0/np.dot(weights,eta0)
        return eta0
        
        
    def weights(self):
        return np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
        
        
    def _minimize(self,tol = 1e-10):
        weights = self.weights()
        eta0 = self.random_eta_short()
        constraints=({'type':'eq','fun':lambda eta: np.dot(eta,weights)-1.0})
        res = minimize(self.m_rep,eta0,bounds = [(0.0,1.0)]*len(eta0), constraints = constraints ,method='SLSQP',tol=tol)      
        return res
            
    def find_eta(self,tol=1e-10):
        eta,m = Model.find_eta(self)
        eta_full = self.expand_eta(eta)
        return eta_full,m 
        

    def m_rep(self,eta_short_form):
        eta = self.expand_eta(eta_short_form)
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def expand_eta(self,eta_short_form):
        arrays = []
        for indx, count in enumerate(self.weights()):
            arrays.append(np.full(count,eta_short_form[indx]))
        return np.hstack(arrays)
        
        
class ParallelConfoundedNoZAction(ParallelConfounded):
    """ the ParallelConfounded Model but without the actions that set Z """
    def __init__(self,q10,q11,q20,q21,pZ,N1,N2,epsilon):
        self._init_pre_action(q10,q11,q20,q21,pZ,N1,N2,epsilon)
        self.K = 2*self.N + 1        
        self.pre_compute() 
              
    def P(self,x):
        p = ParallelConfounded.P(self,x)
        return np.hstack((p[0:-3],p[-1]))        
        
        
    def sample(self,action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        z = binomial(1,self.pZ)        
        x = binomial(1,self.pXgivenZ[1,:,z]) # PXgivenZ[j,i,k] = P(X_i=j|Z=k)
        
        if action < 2*self.N: # setting x_i = j
             i,j = action % self.N, action/self.N
             x[i] = j
             
        y = binomial(1,self.pYgivenX(x)) - self.get_costs()[action]
        
        return x,y
          
    def weights(self):
        return np.asarray([self.N1,self.N2,self.N1,self.N2,1])
              

    
        
        

        
        
        
    



if __name__ == "__main__":  
    N = 8
    N1 = 1
    pz = .5
    q = (1,1,1,0)
    epsilon = .1
    simulations = 10
   
    model = ParallelConfounded.create(N,N1,pz,q,epsilon)
    
    
    
  
    
   
    
    
    

















    
