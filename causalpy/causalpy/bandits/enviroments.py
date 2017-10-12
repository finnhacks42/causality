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
from itertools import product,chain
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
    
class ProbabilityConstraintError(ValueError):
    '''Thrown when probabilities are negative or do not sum to 1'''
    

    

    
class DiscreteContextualEnviroment(object):
    def __init__(self,reward_matrix, context_distribution):
        """
        reward_matrix: 2d array containing the probabilities of obtaining a (binary) reward 
                       with a column for each context and row for each action.
        context_distribution: a vector containing the probability that each context is selected.
        """
        self.num_contexts = reward_matrix.shape[1]
        self.K = reward_matrix.shape[0]
        self.contexts = np.arange(self.num_contexts)
        self.reward_matrix = reward_matrix 
        self.context_distribution = context_distribution
        
        # expected marginal rewards
        self.expected_rewards = np.dot(self.reward_matrix,self.context_distribution) 
        
        
    @classmethod
    def create_random(cls,num_contexts,num_actions):
        reward_matrix = np.random.uniform(size=(num_actions,num_contexts))
        context_distribution = np.full(num_contexts,1.0/num_contexts) 
        return cls(reward_matrix,context_distribution)
        
    @classmethod
    def create_uninformative_context(cls,num_contexts,num_actions,epsilon = 0.1):
        reward_dist = np.full(num_actions,.5)
        reward_dist[0] = .5+epsilon
        reward_dist = reward_dist.reshape((num_actions,1))
        reward_matrix = np.tile(reward_dist,num_contexts)# same reward distribution for each context.
        context_distribution = np.full(num_contexts,1.0/num_contexts) 
        return cls(reward_matrix,context_distribution)
                
    def sample_context(self):
        """ Sample a value from the context """
        return np.random.choice(self.contexts,p=self.context_distribution)
        
    def best_reward(self):
        """ return a vector containing reward of the optimal action for each context."""
        return self.reward_matrix.max(axis=0)
        
    def expected_reward_sequence(self,arms,contexts):
        """ 
        arms: a vector containing which arm was pulled in each timestep.
        contexts: a vector of the corresponding contexts.
        returns: a vector of the expected rewards at each timestep. 
        """
        return self.reward_matrix[arms,contexts]
        
    
    def expected_conditional_rewards(self,context):
        """
        Returns a vector of the expected rewards for all the actions given the supplied context.
        """
        return self.reward_matrix[:,context]
    
    def sample_conditional_reward(self,context,action,n=1):
        """
        Sample reward(s) from P(Y = 1|C,A)
        returns a 1-d array containing the samples
        """
        return np.random.binomial(1,self.reward_matrix[action,context],size=n)
    
    def sample_conditional_rewards(self,context,n=1):
        """
        Sample the rewards for all the actions given the supplied context.
        returns a 2d-array where each row is a sample and each column corresponds to an action. 
        """
        return np.random.binomial(1,self.reward_matrix[:,context],size=(n,self.num_contexts))
        
        # methods for algorithms that are unaware of the context
    def sample_multiple(self,actions,n=1):
        """ sample from the marginal expected rewards """
        return binomial(n,self.expected_rewards[actions])
    
    
class TimeVaryingBernoulliModel(object):
    """ 
    A model that shifts the rewards for each arm linearly,
    as a function of the time-step".
    """
    def __init__(self,initial_expected_rewards,final_expected_rewards,T):
        self.K = len(initial_expected_rewards)
        self.t = 0
        self.slope = (final_expected_rewards - initial_expected_rewards)/float(T)
        self.intercept = initial_expected_rewards
        self.expected_rewards = self.expected_rewards_t(0)
    
    def expected_rewards_t(self,t):
        return self.intercept + self.slope*t
        
    def sample_multiple(self,actions,n):
        p = self.expected_rewards[actions]
        self.t += 1
        self.expected_rewards = self.expected_rewards_t(self.t)
        return binomial(n,p)
    
class BasicBernoulliModel(object):
    """ A model for generating data for standard bandits without side information """
    
    def __init__(self,expected_rewards):
        """
        expected_rewards: an array of length K containing the expected rewards for each arm.
        """
        self.expected_rewards = np.asarray(expected_rewards)
        self.K = len(self.expected_rewards)
                
    @classmethod
    def create_epsilon_best(cls,K,epsilon):
        """ creates a bandit where all arms have reward .5 except the last, which has .5 + epsilon """
        expected_rewards = np.full(K,.5)
        expected_rewards[-1] = .5 + epsilon # reward of one arm is just better than the others.
        return cls(expected_rewards)
        
    @classmethod
    def create_random(cls,K):
        """ creates a bandit where the expected rewards are sampled from a uniform distribution """
        expected_rewards = np.random.uniform(size=K)
        return cls(expected_rewards)
    
    @classmethod
    def create_worst_case(cls,K,T):
        epsilon = np.sqrt(K/(4.0*T))
        return cls.create_epsilon_best(K,epsilon)
        
    
    def sample_multiple(self,actions,n):
        """ draws n samples from the reward distributions of the specified actions. """
        return binomial(n,self.expected_rewards[actions])

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
        assert max(self.expected_rewards) <= 1
        assert min(self.expected_rewards) >= 0
        
    def make_ith_arm_epsilon_best(self,epsilon,i):
        """ adjusts the costs such that all arms have expected reward .5, expect the first one which has reward .5 + epsilon """
        costs = self.expected_Y - 0.5
        costs[i] -= epsilon
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
        
    def _minimize(self,tol,options):
        eta0 = self.random_eta()
        constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0})
        #options={'disp': True}
        res = minimize(self.m_eta, eta0,bounds = [(0.0,1.0)]*self.K, constraints = constraints, method='SLSQP',options = options)
        return res
        
    def find_eta(self,tol = 1e-10,min_starts = 1, max_starts = 10,  options={'disp': True, 'maxiter':200}):
        m = self.K + 1
        eta = None
        starts = 0
        success = 0
        while success < min_starts and starts < max_starts:
            res = self._minimize(tol,options)            
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
        
    def action_tuple(self,action):
        """ convert from action id to the tuple (varaible,value) """
        if action == 2*self.N+1:
            return ((None,None))
        return (action % self.N, action/self.N)
        
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
    
    def __init__(self,pZ,pXgivenZ,pYfunc):
        self._init_pre_action(pZ,pXgivenZ,pYfunc,3)
        self.pre_compute()  
        
    def _init_pre_action(self,pZ,pXgivenZ,pYfunc,num_non_x_actions):
        """ The initialization that should occur regardless of whether we can act on Z """
        self.N = pXgivenZ.shape[1]
        self.indx = np.arange(self.N)
        self.pZ = pZ
        self.pXgivenZ = pXgivenZ # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
        self.pYfunc = pYfunc
        #self.pytable = pY #np.asarray([[.4,.4],[.7,.7]])  
        
        # variables X for which pXgivenZ is identical must have the same value for eta.
        group_values = []
        self.group_members = [] #variables in each group
        for var in range(self.N):
            matched = False
            value = self.pXgivenZ[:,var,:]
            for group,gv in enumerate(group_values):
                if np.allclose(value,gv):
                    self.group_members[group].append(var)
                    matched = True
                    break
            if not matched:
                group_values.append(value)
                self.group_members.append([var])
        counts = [len(members) for members in self.group_members]
        self.group_members = [np.asarray(members,dtype=int) for members in self.group_members]
        
        self.weights = list(chain(counts*2,[1]*num_non_x_actions))
        self.K = 2*self.N + num_non_x_actions
        self.nnx = num_non_x_actions
        
    @classmethod
    def pY_epsilon_best(cls,q,pZ,epsilon):
        """ returns a table pY with Y epsilon-optimal for X1=1, sub-optimal for X1=0 and .5 for all others"""
        q10,q11,q20,q21 = q
        px1 = (1-pZ)*q10+pZ*q11
        px0 = (1-pZ)*(1-q10)+pZ*(1-q11)              
        epsilon2 = (px1/px0)*epsilon
        assert epsilon2 < .5
        pY = np.asarray([[.5-epsilon2,.5-epsilon2],[.5+epsilon,.5+epsilon]])         
        return pY
        
               
    @classmethod
    def create(cls,N,N1,pz,pY,q):
        """ builds ParallelConfounded model"""
        q10,q11,q20,q21 = q
        N2 = N - N1
        pXgivenZ0 = np.hstack((np.full(N1,q10),np.full(N2,q20)))
        pXgivenZ1 = np.hstack((np.full(N1,q11),np.full(N2,q21)))
        pX0 = np.vstack((1.0-pXgivenZ0,pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        pX1 = np.vstack((1.0-pXgivenZ1,pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        pXgivenZ = np.stack((pX0,pX1),axis=2) # PXgivenZ[i,j,k] = P(X_i=j|Z=k)
        pYfunc = lambda x: pY[x[0],x[N-1]]
        model = cls(pz,pXgivenZ,pYfunc)
        return model
        
        
    def pYgivenX(self,x):
        return self.pYfunc(x)
        
    def action_tuple(self,action):
        """ convert from action id to the tuple (varaible,value) """
        if action == 2*self.N+1:
            return ('z',1)
        if action ==  2*self.N:
            return ('z',0)
        if action == 2*self.N+2:
            return ((None,None))
        return (action % self.N, action/self.N)
     
             
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
        
        
    def _minimize(self,tol,options):
        eta0 = np.random.random(len(self.group_members)*2+self.nnx)
        eta0 = eta0/np.dot(self.weights,eta0)
        
        constraints=({'type':'eq','fun':lambda eta: np.dot(eta,self.weights)-1.0})
        res = minimize(self.m_rep,eta0,bounds = [(0.0,1.0)]*len(eta0), constraints = constraints ,method='SLSQP',tol=tol,options=options)      
        return res
            
    def find_eta(self,tol=1e-10):
        eta,m = Model.find_eta(self)
        self.eta_short = eta
        eta_full = self.expand(eta)
        return eta_full,m 
        
    def m_rep(self,eta_short_form):
        eta = self.expand(eta_short_form)
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def expand(self,short_form): # not quite right
        # short form is group1=0,group2=0,group3=0,...,group1=1,...group
        eta_full = np.zeros(self.K)
        eta_full[-self.nnx:] = short_form[-self.nnx:]
        num_groups = len(self.group_members)
        for group,members in enumerate(self.group_members):
            eta0 = short_form[group]
            eta1 = short_form[num_groups+group]
            eta_full[members] = eta0
            eta_full[members+self.N] = eta1
    
#        arrays = []
#        for indx, count in enumerate(self.weights()):
#            arrays.append(np.full(count,short_form[indx]))
#        result = np.hstack(arrays)
        return eta_full
        
#    def contract(self,long_form):
#        result = np.zeros(7)
#        result[0] = long_form[0]
#        result[1] = long_form[self.N-1]
#        result[2] = long_form[self.N]
#        result[3] = long_form[2*self.N-1]
#        result[4:] = long_form[-3:]
#        return result
        
        
        
class ParallelConfoundedNoZAction(ParallelConfounded):
    """ the ParallelConfounded Model but without the actions that set Z """
    def __init__(self,pZ,pXgivenZ,pYfunc):
        self._init_pre_action(pZ,pXgivenZ,pYfunc,1)
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
          
#    def weights(self):
#        return np.asarray([self.N1,self.N2,self.N1,self.N2,1])
#        
#    def contract(self,long_form):
#        result = np.zeros(5)
#        result[0] = long_form[0]
#        result[1] = long_form[self.N-1]
#        result[2] = long_form[self.N]
#        result[3] = long_form[2*self.N-1]
#        result[4] = long_form[-1]
#        return result
              

    
              
class ScaleableParallelConfounded(Model):
    """ Makes use of symetries to avoid exponential combinatorics in calculating V """
    # do(x1=0),do(x2=0),do(x1=1),do(x2=1),do(z=0),do(z=1),do()
        
    def __init__(self,q,pZ,pY,N1,N2,compute_m = True):       
        self._init_pre_action(q,pZ,pY,N1,N2,compute_m = True,num_nonx_actions=3)
            
    def _init_pre_action(self,q,pZ,pY,N1,N2,compute_m,num_nonx_actions):
        q10,q11,q20,q21 = q
        self.N = N1+N2
        self.indx = np.arange(self.N)
        self.N1,self.N2 = N1,N2
        self.q = q
        self.pZ = pZ
        self.pytable = pY
        self.pZgivenA = np.hstack((np.full(4,pZ),0,1,pZ))
        pXgivenZ0 = np.hstack((np.full(N1,q10),np.full(N2,q20)))
        pXgivenZ1 = np.hstack((np.full(N1,q11),np.full(N2,q21)))
        pX0 = np.vstack((1.0-pXgivenZ0,pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        pX1 = np.vstack((1.0-pXgivenZ1,pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        self.pXgivenZ = np.stack((pX0,pX1),axis=2) # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
        self.K = 2*self.N + num_nonx_actions
        self.qz0 = np.asarray([(1-q10),q10,(1-q20),q20])
        self.qz1 = np.asarray([(1-q11),q11,(1-q21),q21])
        self._compute_expected_reward()
        
        if compute_m:
            self.compute_m()
            
    def compute_m(self,eta_short = None):
        if eta_short is not None:
            self.m = max(self.V_short(eta_short))
            self.eta = self.expand(eta_short)
        else:
            self.eta,self.m = self.find_eta()
            
    def pYgivenX(self,x):
        i,j = x[0],x[self.N-1]
        return self.pytable[i,j]
        
    def _compute_expected_reward(self):
        q10,q11,q20,q21 = self.q
        pz = self.pZ
        a,b,c,d = self.pytable[0,0],self.pytable[0,1],self.pytable[1,0],self.pytable[1,1]
        alpha = (1-pz)*(1-q10)*(1-q20)+pz*(1-q11)*(1-q21)
        beta = (1-pz)*(1-q10)*q20+pz*(1-q11)*q21
        gamma = (1-pz)*q10*(1-q20)+pz*q11*(1-q21)
        delta = (1-pz)*q10*q20+pz*q11*q21
        dox10 = a*((1-pz)*(1-q20)+pz*(1-q21)) + b*((1-pz)*q20+pz*q21)
        dox11 = c*((1-pz)*(1-q20)+pz*(1-q21)) + d*((1-pz)*q20+pz*q21)
        dox20 = a*((1-pz)*(1-q10)+pz*(1-q11))+c*((1-pz)*q10+pz*q11)
        dox21 = b*((1-pz)*(1-q10)+pz*(1-q11))+d*((1-pz)*q10+pz*q11)
        doxj = a*alpha+b*beta+c*gamma+d*delta
        doz0 = a*(1-q10)*(1-q20)+b*(1-q10)*q20+c*q10*(1-q20)+d*q10*q20
        doz1 = a*(1-q11)*(1-q21)+b*(1-q11)*q21+c*q11*(1-q21)+d*q11*q21
        self.expected_Y = np.hstack((dox10,np.full(self.N-2,doxj),dox20,dox11,np.full(self.N-2,doxj),dox21,doz0,doz1,doxj))
        self.expected_rewards = self.expected_Y
        
    def P(self,x):
        n1,n2 = x[0:self.N1].sum(),x[self.N1:].sum()
        pz0,pz1 = self.p_n_given_z(n1,n2)
        pc = self.pZgivenA*pz1+(1-self.pZgivenA)*pz0
        doxi0 = np.hstack((np.full(self.N1,pc[0]),np.full(self.N2,pc[1])))
        doxi1 = np.hstack((np.full(self.N1,pc[2]),np.full(self.N2,pc[3])))
        pij = np.vstack((doxi0,doxi1))
        pij[1-x,self.indx] = 0
        pij = pij.reshape((self.N*2,))
        result = np.hstack((pij,pc[4],pc[5],pc[6]))
        return result
        
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
        
        
    def V_short(self,eta):
        sum0 = np.zeros(7,dtype=float)
        sum1 = np.zeros(7,dtype=float)
        for n1,n2 in product(range(self.N1+1),range(self.N2+1)):
             wdo = comb(self.N1,n1,exact=True)*comb(self.N2,n2,exact=True)
             wdox10 = comb(self.N1-1,n1,exact=True)*comb(self.N2,n2,exact=True)
             wdox11 = comb(self.N1-1,n1-1,exact=True)*comb(self.N2,n2,exact=True)
             wdox20 = comb(self.N1,n1,exact=True)*comb(self.N2-1,n2,exact=True)
             wdox21 = comb(self.N1,n1,exact=True)*comb(self.N2-1,n2-1,exact=True)
             w = np.asarray([wdox10,wdox20,wdox11,wdox21,wdo,wdo,wdo])
             
             pz0,pz1 = self.p_n_given_z(n1,n2)

             counts = [self.N1-n1,self.N2-n2,n1,n2,1,1,1]
             Q = (eta*pz0*counts*(1-self.pZgivenA)+eta*pz1*counts*self.pZgivenA).sum()
             
             ratio = np.nan_to_num(np.true_divide(pz0*(1-self.pZgivenA)+pz1*self.pZgivenA,Q))
          
             sum0 += np.asfarray(w*pz0*ratio)
             sum1 += np.asfarray(w*pz1*ratio)
        result = self.pZgivenA*sum1+(1-self.pZgivenA)*sum0
        return result
        
    def V(self,eta):
        eta_short_form = self.contract(eta)
        v = self.V_short(eta_short_form)
        v_long = self.expand(v)
        return v_long
        
    def m_rep(self,eta_short_form):
        V = self.V_short(eta_short_form)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
    
    def find_eta(self,tol=1e-10):
        eta,m = Model.find_eta(self)
        self.eta_short = eta
        eta_full = self.expand(eta)
        return eta_full,m 
        
    def _minimize(self,tol,options):
        weights = self.weights()
        eta0 = self.random_eta_short()
        constraints=({'type':'eq','fun':lambda eta: np.dot(eta,weights)-1.0})
        res = minimize(self.m_rep,eta0,bounds = [(0.0,1.0)]*len(eta0), constraints = constraints ,method='SLSQP',tol=tol,options=options)      
        return res
    
    def weights(self):
        return np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
                   
            
    def p_n_given_z(self,n1,n2):
        powers = np.tile([self.N1-n1,n1,self.N2-n2,n2],7).reshape((7,4))
        powers[0,0]-=1 #do(x1=0)
        powers[1,2]-=1 #do(x2=0)
        powers[2,1]-=1 #do(x1=1)
        powers[3,3]-=1 #do(x2=1)
        
        pnz0 = (self.qz0**powers).prod(axis=1)
        pnz1 = (self.qz1**powers).prod(axis=1)
        return pnz0,pnz1
        
    def random_eta_short(self):
        weights = self.weights()
        eta0 = np.random.random(len(weights))
        eta0 = eta0/np.dot(weights,eta0)
        return eta0
        
    def contract(self,long_form):
        result = np.zeros(7)
        result[0] = long_form[0]
        result[1] = long_form[self.N-1]
        result[2] = long_form[self.N]
        result[3] = long_form[2*self.N-1]
        result[4:] = long_form[-3:]
        return result
        
    def expand(self,short_form):
        arrays = []
        for indx, count in enumerate(self.weights()):
            arrays.append(np.full(count,short_form[indx]))
        result = np.hstack(arrays)
        return result
        
    

class ScaleableParallelConfoundedNoZAction(ScaleableParallelConfounded):
      
    def __init__(self,q,pZ,pY,N1,N2,compute_m = True):
        self._init_pre_action(q,pZ,pY,N1,N2,compute_m,1)
        self.expected_rewards = self._mask(self.expected_rewards)
        self.expected_Y = self._mask(self.expected_Y)

    def _mask(self,vect):
        return np.hstack((vect[0:-3],vect[-1]))
        
    
    def P(self,x):
        p = ScaleableParallelConfounded.P(self,x)
        return self._mask(p)
    
    def V(self,eta):
        eta_short_form = self.contract(eta)
        eta = np.hstack((eta_short_form[0:-1],0,0,eta_short_form[-1]))
        v = self.V_short(eta) # length 7
        v = self._mask(v) # length 5
        v_long = self.expand(v)
        return v_long
        
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
        
    
        
    def m_rep(self,eta_short_form):
        eta = np.hstack((eta_short_form[0:-1],0,0,eta_short_form[-1]))
        V = self.V_short(eta)
        V[-3:-1] = 0 # exclude do(z=0) and do(z=1)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def contract(self,long_form):
        result = np.zeros(5)
        result[0] = long_form[0]
        result[1] = long_form[self.N-1]
        result[2] = long_form[self.N]
        result[3] = long_form[2*self.N-1]
        result[4] = long_form[-1]
        return result


def estimate_px_and_y_from_samples(model,samples):
        expected_y = np.zeros(model.K,dtype=float)
        shape = list(chain([model.K],[2]*model.N))
        xcounts = np.zeros(shape)
        for a in range(model.K):
            for s in xrange(samples):
                x,y = model.sample(a)
                pos = tuple(chain([a],x))
                
                xcounts[pos] +=1
               
                expected_y[a] += y
            expected_y[a] = expected_y[a]/samples
        xcounts = xcounts/samples
        return xcounts,expected_y
        
