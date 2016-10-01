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
from scipy.stats import binom
from scipy.misc import comb


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
        
    def make_ith_arm_epsilon_best(self,epsilon,i):
        """ adjusts the costs such that all arms have expected reward .5, expect the first one which has reward .5 + epsilon """
        # TODO not clear that truncation in general is correct given this        
        #costs = self.expected_Y - 0.5
        #costs[i] -= epsilon
        #self.set_action_costs(costs)
        
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
        
        self.eta,self.m = self.find_eta()
        self.eta = self.eta/self.eta.sum() # random choice demands more accuracy than contraint in minimizer
        
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
        
class ActionShuffler(object):
    """ takes an input model and re-orders all the actions """    
    def __init__(self,model):
        self.model = model
        self.indx = np.arange(model.K)
        np.random.shuffle(self.indx) 
        self.m = model.m
        self.K = model.K
        self.eta = model.eta[self.indx]
        self.expected_rewards = model.expected_rewards[self.indx]
        
    
    def get_costs(self):
        return self.model.get_costs()[self.indx]
        
    def P(self,x):
        return self.model.P(x)[self.indx]
        
    def sample_multiple(self,actions,n):
        return self.model.sample_multiple(self.indx[actions])
        
        


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
        y = binomial(1,self.pYgivenX(x))
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
    
    def __init__(self,q,pZ,pY,N1,N2,epsilon):
        self._init_pre_action(q,pZ,pY,N1,N2,epsilon)
        self.K = 2*self.N + 3        
        self.pre_compute()  
        
    def _init_pre_action(self,q,pZ,pY,N1,N2,epsilon):
        """ The initialization that should occur regardless of whether we can act on Z """
        self.N1 = N1
        self.N2 = N2
        self.q10,self.q11,self.q20,self.q21 = q
        self.N = N1+N2
        self.indx = np.arange(self.N)
        self.pZ = pZ
        
        pXgivenZ0 = np.hstack((np.full(N1,self.q10),np.full(N2,self.q20)))
        pXgivenZ1 = np.hstack((np.full(N1,self.q11),np.full(N2,self.q21)))
        self.pX0 = np.vstack((1.0-pXgivenZ0,pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        self.pX1 = np.vstack((1.0-pXgivenZ1,pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        self.pXgivenZ = np.stack((self.pX0,self.pX1),axis=2) # PXgivenZ[i,j,k] = P(X_i=j|Z=k)
    
        
        self.pytable = pY#np.asarray([[.4,.4],[.7,.7]])   
    
    def __str__(self):
        string = "ParallelConfounded_mis{0:.1f}_Nis{1}_N1is{2}_qis{3:.1f}_{4:.1f}_{5:.1f}_{6:.1f}_pzis{7:.1f}_epsilonis{8:.1f}".format(self.m,self.N,self.N1,self.q10,self.q11,self.q20,self.q21,self.pZ,self.epsilon)
        return string.replace(".","-")
        
               
    @classmethod
    def create(cls,N,N1,pz,pY,q,epsilon):
        """ builds ParallelConfounded model"""
        q10,q11,q20,q21 = q
        N2 = N - N1
        model = cls(q,pz,pY,N1,N2,epsilon)
        return model
        
        
    def pYgivenX(self,x):
        i,j = x[0],x[self.N-1]
        return self.pytable[i,j] 
     
             
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
             
        y = binomial(1,self.pYgivenX(x)) 
        
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
    def __init__(self,q,pZ,pY,N1,N2,epsilon):
        self._init_pre_action(q,pZ,pY,N1,N2,epsilon)
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
             
        y = binomial(1,self.pYgivenX(x)) 
        
        return x,y
          
    def weights(self):
        return np.asarray([self.N1,self.N2,self.N1,self.N2,1])
              
              
class ScaleableParallelConfounded():
    """ Makes use of symetries to avoid exponential combinatorics in calculating V """
    # do(x1=0),do(x2=0),do(x1=1),do(x2=1),do(z=0),do(z=1),do()
    
    def __init__(self,q,pZ,pY,N1,N2):
        self.pZ = pZ
        self.pZgivenA = np.hstack((np.full(4,pZ),0,1,pZ))
        self.N1 = N1
        self.N2 = N2
        self.N = N1+N2
        self.K = 2*self.N + 3
        q10,q11,q20,q21 = q
        self.q10,self.q11,self.q20,self.q21 = q
        self.qz0 = np.asarray([(1-q10),q10,(1-q20),q20])
        self.qz1 = np.asarray([(1-q11),q11,(1-q21),q21])
        self.pa = np.zeros(7)
        self.indx = np.arange(self.N)
        self.weights = np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
         
    def P(self,x):
        n1 = x[0:self.N1].sum()
        n2 = x[self.N1:].sum()
        pc = self.P_counts(n1,n2)
        doxi0 = np.hstack((np.full(self.N1,pc[0]),np.full(self.N2,pc[1])))
        doxi1 = np.hstack((np.full(self.N1,pc[2]),np.full(self.N2,pc[3])))
        pij = np.vstack((doxi0,doxi1))
        pij[1-x,self.indx] = 0
        pij = pij.reshape((self.N*2,))
        result = np.hstack((pij,pc[4],pc[5],pc[6]))
        return result
        
    def P02(self,x):
        
        pXgivenZ0 = np.hstack((np.full(self.N1,self.q10),np.full(self.N2,self.q20)))
        pXgivenZ0 = np.vstack((1-pXgivenZ0,pXgivenZ0))

        
        pz0 = pXgivenZ0[x,self.indx]
    
        p_obs = pz0.prod()
        
        # for do(x_i = j)
        p = prod_all_but_j(pz0) # vector of length N
           
        pij = np.vstack((p,p))
        pij[1-x,self.indx] = 0 # 2*N array, pij[i,j] = P(X=x|do(X_i=j)) = d(X_i-j)*prod_k!=j(X_k = x_k)
        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
        
        result = np.hstack((pij,pz0.prod(),0,p_obs))
        return result
        
    
    def P0(self,x):
        n1 = x[0:self.N1].sum()
        n2 = x[self.N1:].sum()
        pc = self.P_counts0(n1,n2)
        doxi0 = np.hstack((np.full(self.N1,pc[0]),np.full(self.N2,pc[1])))
        doxi1 = np.hstack((np.full(self.N1,pc[2]),np.full(self.N2,pc[3])))
        pij = np.vstack((doxi0,doxi1))
        pij[1-x,self.indx] = 0
        pij = pij.reshape((self.N*2,))
        result = np.hstack((pij,pc[4],pc[5],pc[6]))
        return result
        
    def P1(self,x):
        n1 = x[0:self.N1].sum()
        n2 = x[self.N1:].sum()
        pc = self.P_counts1(n1,n2)
        doxi0 = np.hstack((np.full(self.N1,pc[0]),np.full(self.N2,pc[1])))
        doxi1 = np.hstack((np.full(self.N1,pc[2]),np.full(self.N2,pc[3])))
        pij = np.vstack((doxi0,doxi1))
        pij[1-x,self.indx] = 0
        pij = pij.reshape((self.N*2,))
        result = np.hstack((pij,pc[4],pc[5],pc[6]))
        return result
        
    def pcz0(self,n1,n2):
        result = np.zeros(7)
        result[0] = binom(self.N1-1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2)
        result[1] = binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2)
        result[2] = binom(self.N1-1,self.q10).pmf(n1-1)*binom(self.N2,self.q20).pmf(n2)
        result[3] = binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2-1)
        result[4] = binom(self.N1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2)
        result[5] = 0
        result[6] = result[4]
        return result
        
    def pcz1(self,n1,n2):
        result = np.zeros(7)
        result[0] = binom(self.N1-1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2)
        result[1] = binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2)
        result[2] = binom(self.N1-1,self.q11).pmf(n1-1)*binom(self.N2,self.q21).pmf(n2)
        result[3] = binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2-1)
        result[4] = 0
        result[5] = binom(self.N1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2)
        result[6] = result[5]
        return result
                
    def pc(self,n1,n2):
        return self.pZgivenA*self.pcz1(n1,n2)+(1-self.pZgivenA)*self.pcz0(n1,n2)
      
    def counts(self):
        return product(range(self.N1+1),range(self.N2+1))
    
    def V_short3(self,eta):
        sum0 = np.zeros(7)
        sum1 = np.zeros(7)
        for n1,n2 in product(range(self.N1+1),range(self.N2+1)):
            pz0 = self.pcz0(n1,n2)
            Q0 = (eta*self.weights).dot(pz0)
            sum0 += pz0*np.nan_to_num(np.true_divide(pz0,Q0))
            pz1 = self.pcz1(n1,n2)
            Q1 = (eta*self.weights).dot(pz1)
            sum1 +=  pz1*np.nan_to_num(np.true_divide(pz1,Q1))
            
        result = (1-self.pZgivenA)*sum0+self.pZgivenA*sum1
        return result
        
    def V_short(self,eta):
        result = np.zeros(7)
        for n1,n2 in product(range(self.N1+1),range(self.N2+1)):
            pa = self.p_of_count_given_action(n1,n2)
            Q = (eta*self.weights).dot(pa)
            ratio = np.nan_to_num(np.true_divide(pa,Q))
            
            result += pa*ratio 
            # it seems like we can only change the muliplier here
            # changing pa earlier changes Q, which changes results for answers that are correct already ...
        return result
        
    
    def p_of_count_given_action(self,n1,n2):
        pa = self.P_counts(n1,n2)
        wdo = comb(self.N1,n1,exact=True)*comb(self.N2,n2,exact=True)
        wdox10 = comb(self.N1-1,n1,exact=True)*comb(self.N2,n2,exact=True)
        wdox11 = comb(self.N1-1,n1-1,exact=True)*comb(self.N2,n2,exact=True)
        wdox20 = comb(self.N1,n1,exact=True)*comb(self.N2-1,n2,exact=True)
        wdox21 = comb(self.N1,n1,exact=True)*comb(self.N2-1,n2-1,exact=True)
        w = np.asarray([wdox10,wdox20,wdox11,wdox21,wdo,wdo,wdo])
        #print (n1,n2),"weight",w,pa
        return w*pa
        
    def pos_power(self,a,b):
        result = a**b
        neg = np.where(b<0)[0]
        result[neg] = 0
        return result
        
        
    def p_of_x_given_a_z(self,x,z):
        n1,n2 = x[0:self.N1].sum(),x[self.N1:].sum()
#        print x,n1,n2
        powers = np.tile([self.N1-n1,n1,self.N2-n2,n2],7).reshape((7,4))
        powers[0,0]-=1 #do(x1=0)
        powers[1,2]-=1 #do(x2=0)
        powers[2,1]-=1 #do(x1=1)
        powers[3,3]-=1 #do(x2=1)
        
        if z == 0:
            return self.pcz0(n1,n2)
            #return self.pos_power(self.qz0,powers).prod(axis=1)
        else:
            return self.pcz1(n1,n2)
            #return self.pos_power(self.qz1,powers).prod(axis=1)
    
    
    def V_short2(self,eta):
        sum0 = np.zeros(7)
        sum1 = np.zeros(7)
        for x in Model.generate_binary_assignments(self.N):
            pz0 = self.p_of_x_given_a_z(x,0)            
            Q0 = (eta*self.weights).dot(pz0)
            sum0 += pz0*np.nan_to_num(np.true_divide(pz0,Q0))
            
            pz1 = self.p_of_x_given_a_z(x,1)
            Q1 = (eta*self.weights).dot(pz1)
            sum1 += pz1*np.nan_to_num(np.true_divide(pz1,Q1))
    
        result = self.pZgivenA*sum1+(1-self.pZgivenA)*sum0
        return result 
        
    def V(self,eta):
        sum0 = np.zeros(self.K)
        sum1 = np.zeros(self.K)
        for x in Model.generate_binary_assignments(self.N):
            pz0 = self.P0(x)            
            Q0 = eta.dot(pz0)
            sum0 += self.P(x)*np.nan_to_num(np.true_divide(pz0,Q0))
            
            pz1 = self.P1(x)
            Q1 = eta.dot(pz1)
            sum1 += self.P(x)*np.nan_to_num(np.true_divide(pz1,Q1))
    
        result = self.expand(self.pZgivenA)*sum1+self.expand(1-self.pZgivenA)*sum0
        return result 
            
  
    def P_counts0(self,n1,n2):
        """ joint probability of variables not set for each action """
        counts = np.asarray([(self.N1-n1),n1,(self.N2-n2),n2]) 
        self.pa[4] = np.power(self.qz0,counts).prod() #do(z=0)
        self.pa[5] = 0 #do(z=1)
        self.pa[6] = self.pa[4] #do()
        
        counts[0] = (self.N1-n1) - 1
        self.pa[0] = np.power(self.qz0,counts).prod() #do(x1=0)
        
        counts[0],counts[2] = (self.N1-n1), (self.N2-n2) - 1
        self.pa[1] = np.power(self.qz0,counts).prod() #do(x2=0)
        
        counts[2],counts[1] = (self.N2-n2), n1-1
        self.pa[2] = np.power(self.qz0,counts).prod()#do(x1=1)
        
        counts[1],counts[3] = n1,n2-1
        self.pa[3]= np.power(self.qz0,counts).prod()#do(x2=1)
        return self.pa
        
    def P_counts1(self,n1,n2):
        """ joint probability of variables not set for each action """
        counts = np.asarray([(self.N1-n1),n1,(self.N2-n2),n2]) 
        self.pa[4] = 0 #do(z=0)
        self.pa[5] = np.power(self.qz1,counts).prod() #do(z=1)
        self.pa[6] = self.pa[5]#do()
        
        counts[0] = (self.N1-n1) - 1
        self.pa[0] = np.power(self.qz1,counts).prod() #do(x1=0)
        
        counts[0],counts[2] = (self.N1-n1), (self.N2-n2) - 1
        self.pa[1] = np.power(self.qz1,counts).prod() #do(x2=0)
        
        counts[2],counts[1] = (self.N2-n2), n1-1
        self.pa[2] = np.power(self.qz1,counts).prod()#do(x1=1)
        
        counts[1],counts[3] = n1,n2-1
        self.pa[3]= np.power(self.qz1,counts).prod()#do(x2=1)
        return self.pa
        
        
    
    def P_counts(self,n1,n2):
        """ joint probability of variables not set for each action """
        counts = np.asarray([(self.N1-n1),n1,(self.N2-n2),n2]) 
        self.pa[4] = np.power(self.qz0,counts).prod() #do(z=0)
        self.pa[5] = np.power(self.qz1,counts).prod() #do(z=1)
        self.pa[6] = self.pa[4]*(1-self.pZ)+self.pa[5]*self.pZ #do()
        
        counts[0] = (self.N1-n1) - 1
        self.pa[0] = np.power(self.qz0,counts).prod()*(1-self.pZ)+np.power(self.qz1,counts).prod()*self.pZ #do(x1=0)
        
        counts[0],counts[2] = (self.N1-n1), (self.N2-n2) - 1
        self.pa[1] = np.power(self.qz0,counts).prod()*(1-self.pZ)+np.power(self.qz1,counts).prod()*self.pZ #do(x2=0)
        
        counts[2],counts[1] = (self.N2-n2), n1-1
        self.pa[2] = np.power(self.qz0,counts).prod()*(1-self.pZ)+np.power(self.qz1,counts).prod()*self.pZ#do(x1=1)
        
        counts[1],counts[3] = n1,n2-1
        self.pa[3]= np.power(self.qz0,counts).prod()*(1-self.pZ)+np.power(self.qz1,counts).prod()*self.pZ#do(x2=1)
        return self.pa
        
    
        
    def expand(self,short_form):
        arrays = []
        for indx, count in enumerate(self.weights):
            arrays.append(np.full(count,short_form[indx]))
        return np.hstack(arrays)
        
    def contract(self,long_form):
        result = np.zeros(7)
        result[0] = long_form[0]
        result[1] = long_form[self.N-1]
        result[2] = long_form[self.N]
        result[3] = long_form[2*self.N-1]
        result[4:] = long_form[-3:]
        return result
        
    
        
    
            
        

        

            
        

    
    
#    def __init__(self,q,pZ,pY,N1,N2):
#       
#        self.parent_assignments = list(product(range(N1+1),range(N2+1))) # all combinations of vector of length 2 specifying [num x_i: i<= N1 = 1, num x_i: i > N = 1]
#        self.pa_given_assignment = np.asarray([self.P_of_reparameterized_assignment(n1,n2) for (n1,n2) in self.parent_assignments])
#                
#    def P_of_reparameterized_assignment(self,n1,n2):
#         pa = np.zeros(7)
#         pa[0] = (1-self.pZ)*binom(self.N1-1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2)+self.pZ*binom(self.N1-1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2) # do(X1 = 0)
#         pa[1] = (1-self.pZ)*binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2)+self.pZ*binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2) # do(X2 = 0)
#         pa[2] = (1-self.pZ)*binom(self.N1-1,self.q10).pmf(n1-1)*binom(self.N2,self.q20).pmf(n2)+self.pZ*binom(self.N1-1,self.q11).pmf(n1-1)*binom(self.N2,self.q21).pmf(n2) # do(X1=1)
#         pa[3] = (1-self.pZ)*binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2-1)+self.pZ*binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2-1) # do(X2 = 1)
#         pa[4] = binom(self.N1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2) # do(Z = 0)
#         pa[5] = binom(self.N1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2) # do(Z = 1)
#         pa[6] = (1-self.pZ)*pa[4]+self.pZ*pa[5] # do()
#         return pa
#         
#    
#    
#        
#    # group variables for which pXgivenZ0 and pXgivenZ1 are equal
#    def V(self,eta):
#        """ eta should be a vector of length 7, do(X_i: i < N1) = 0, do(X_i: i >= N1) = 0,  do(X_i: i < N1) = 0, do(X_i: i >= N1) = 0 do(Z=0),do(Z=1),do().
#            returns a corresponding vector of length 7"""
#        nom = (self.pa_given_assignment**2).T
#        denom = np.dot(self.pa_given_assignment,eta)
#        r = np.true_divide(nom,denom)
#        return r.sum(axis=1)
#       
#    def unparameterize_eta(self,eta):
#        """ transform 7 dimensional eta back to K dimensional eta """
#        result = np.hstack((np.full(self.N1,eta[0]/self.N1),
#                            np.full(self.N2,eta[1]/self.N2),
#                            np.full(self.N1,eta[2]/self.N1),
#                            np.full(self.N2,eta[3]/self.N2),
#                            eta[4],eta[5],eta[6]))
#        #TODO check if this ordering is correct ...
#        return result
#    
#    def parameterize(x,self):
#        """ transform x into n1,n2 counts """
#        n1 = sum(x[0:self.N1])
#        n2 = sum(x[self.N1:])
#        return n1,n2 

    
        
        

        
        
        
    



if __name__ == "__main__":  
    import numpy.testing as np_test
    N = 3
    N1 = 1
    N2 = N-N1
    q = .1,.3,.4,.7
    q10,q11,q20,q21 = q
    pZ = .2
    pY = np.asanyarray([[.2,.8],[.3,.9]])
    model1 = ParallelConfounded.create(N,N1,pZ,pY,q,.1)
    model2 = ScaleableParallelConfounded(q,pZ,pY,N1,N2)
    
    m = model2  
    
    eta = model1.expand_eta(model1.random_eta_short())
    
    print model1.V(eta)
    print model2.V(eta)
    
    totals0 = np.zeros(7)
    totals1 = np.zeros(7)    
    for n1,n2 in model2.counts():
        totals0 += model2.pcz0(n1,n2)
        totals1 += model2.pcz1(n1,n2)
    print "t",totals0
    print "t",totals1
#    
#    
    totals = np.zeros(model2.K)
    for x in Model.generate_binary_assignments(N):
        totals+=m.P1(x)
    print "t",totals
    
    
    
#    totals = np.zeros(model2.K)
#    for x in Model.generate_binary_assignments(N):
#        totals+=m.P0(x)
#        print m.P0(x)
#        print m.P02(x)
#        print "\n"
#        np_test.assert_almost_equal(m.P0(x),m.P02(x))
#    print totals
#    
    
    
   

    
    
    
    #eta = np.zeros(model1.K)
    #eta[0:N1] = 1.0/2*N1
    #eta[-1] = 1.0/2
    
    
    
    
    
    

#        
#    for n1,n2 in product(range(model2.N1+1),range(model2.N2+1)):
#        print (n1,n2),model2.p_of_count_given_action(n1,n2)[1]
#        
#    print "\n"
#        
#    for x in model1.get_parent_assignments():
#        print x,model2.P(x)[1]
   
    
#    x = np.asarray([0,0,1],dtype=int)
#    n1,n2 = x[0:N1].sum(),x[N1:].sum()
#    p = model1.P(x)
#    p1 = model2.contract(p)
#    p2 = model2.P_counts(n1,n2)
#    
#    print x
#    print n1,n2
#    print p,"\n"
#    
#    print p1
#    print p2
#    
    
   
           
    

        
    
    
  
    
   
    
    
    

















    
