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
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from scipy.optimize import minimize
import time

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
        self.expected_rewards = self.expected_Y - costs
        self.optimal = max(self.expected_rewards)
    
    def pre_compute(self,compute_py = True):
        """ 
        pre-computes expensive results 
        A is an lxk matrix such that A[i,j] = P(ith assignment | jth action)
        PY is an lx1 vector such that PY[i] = P(Y|ith assignment)
        """

        self.generate_binary_assignments()
 
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
        
    
        
    def generate_binary_assignments(self):
        """ generate all possible binary assignments to the N parents of Y. """
        self.parent_assignments = map(np.asarray,product([0,1],repeat = self.N))
        return self.parent_assignments
    
    def R(self,pa,eta):
        """ returns the ratio of the probability of the given assignment under each action to the probability under the eta weighted sum of actions. """
        Q = (eta*pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
               
        
    def V(self,eta):
        """ returns a vector of length K with the expected value of R (over x sampled from p(x|a)) for each action a """
        u = 1.0/np.dot(self.A,eta)
        v = np.dot(self.A2T,u)
        return np.nan_to_num(v)
    
    def m(self,eta):
        """ The maximum value of V"""
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m should not be nan, \n{0}\n{1}".format(eta,V)
        return maxV
        
    def random_eta(self):
        eta = np.random.random(self.K)
        return eta/eta.sum()
        
    def find_eta(self,tol = 1e-6):
        eta0 = self.random_eta()
        constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0})
        res = minimize(self.m, eta0,bounds = [(0.0,1.0)]*self.K, constraints = constraints ,options={'disp': True},method='SLSQP')
        assert res.success, " optimization failed to converge"+res.message
        return res.x,res.fun
             
    def sample_multiple(self,actions,n):
        """ draws n samples from the reward distributions of the specified actions. """
        return binomial(n,self.expected_rewards[actions])
        

        
class GeneralModel(Model):
    """ Allows construction of an arbitray causal graph & action space with discrete (currently assumed binary) CPD tables. 
        This implementation will not scale to large graphs. """
    def __init__(self,model,actions):
        """ model is a pgmpy.BayesianModel
            actions is a list of (var,value) tuples """
        self.parents = sorted(model.get_parents('Y'))
        self.N = len(self.parents)
        self.actions = actions
        self.K = len(actions)
        
        self.post_action_models = [GeneralModel.do(model,action) for action in actions]
        self.samplers = [BayesianModelSampling(model_a) for model_a in self.post_action_models]
        
        self.interventional_distributions = []
        for indx,new_model in enumerate(self.post_action_models):
            infer = VariableElimination(new_model)
            _,distribution_over_parents = infer.query(self.parents)
            self.interventional_distributions.append(distribution_over_parents)        
           
        self.pre_compute(compute_py = False)
        
    def _expected_Y(self):
        expected_Y = np.zeros(self.K)
        for indx,new_model in enumerate(self.post_action_models):
            infer = VariableElimination(new_model)
            _,distribution_over_reward = infer.query(['Y'])
            expected_reward = distribution_over_reward.reduce([('Y',1)],inplace=False).values #TODO investigate failing if inplace=True - bug in pgmpy?
            expected_Y[indx] = expected_reward
        return expected_Y
        
    
    @classmethod
    def create_confounded_parallel(cls,N,N1,pz,q,epsilon):       
        """ convinience method for constructing equivelent models to Confounded_Parallel""" 
        q10,q11,q20,q21 = q
        pZ = [[1-pz,pz]] 
        pXgivenZ_N1 = [[1-q10,1-q11],[q10,q11]] 
        pXgivenZ_N2 = [[1-q20,1-q21],[q20,q21]] 
                                                         
        xvars = ['X'+str(i) for i in range(1,N+1)]
        edges = chain([('Z',v) for v in xvars],[(v,'Y') for v in xvars])
        model = BayesianModel(edges)
        cpds = [TabularCPD(variable='Z',variable_card=2,values=pZ)]
        cpds.extend([TabularCPD(variable=v,variable_card=2,values=pXgivenZ_N1, evidence=['Z'], evidence_card = [2]) for v in xvars[0:N1] ])
        cpds.extend([TabularCPD(variable=v,variable_card=2,values=pXgivenZ_N2, evidence=['Z'], evidence_card = [2]) for v in xvars[N1:] ])
        
        px1 = (1-pz)*q10+pz*q11
        epsilon2 = px1/(1-px1)*epsilon
        pYis1 = np.hstack((np.full(2**(N-1),.5-epsilon2),np.full(2**(N-1),.5+epsilon)))
        ycpd = np.vstack((1-pYis1,pYis1))
        cpds.append(TabularCPD(variable='Y',variable_card=2, values = ycpd, evidence = xvars,evidence_card = [2]*len(xvars)))
        
        model.add_cpds(*cpds)
        model.check_model()
        actions = list(chain([(x,0) for x in xvars],[(x,1) for x in xvars],[("Z",i) for i in (0,1)],[(None,None)]))
        pgm_model = cls(model,actions)  
        # adjust costs for do(Z=1), do(Z=0) such that the actions have expected reward .5 to match worse case 
        costs = np.zeros(pgm_model.K)
        costs[-2] = pgm_model.expected_Y[-2]-.5 
        costs[-3] = pgm_model.expected_Y[-3]-.5
        pgm_model.set_action_costs(costs)
        
        return pgm_model
    
    @classmethod
    def do(cls,model,action):
        var,value = action
        new_model = BayesianModel(model.edges())
        if var is not None:
            for p in model.get_parents(var):
                new_model.remove_edge(p,var)
        cpds = []
        for cpd in model.get_cpds():
            if cpd.variable == var:
                values = np.zeros((cpd.variable_card,1))
                values[value] = 1.0
                values[1-value] = 0.0
                cpd_new = TabularCPD(variable=var,variable_card = cpd.variable_card, values = values)
                cpds.append(cpd_new)
            else:
                cpds.append(cpd.copy())
        new_model.add_cpds(*cpds)
        new_model.check_model()
        return new_model
        
    def sample(self,action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """
        s = self.samplers[action].forward_sample()
        x = s.loc[:,self.parents].values[0]
        y = s.loc[:,['Y']].values[0][0]
        return x,y
       
    def P(self,x):
        """ returns the probability of the given assignment to the parents of Y for given each action. """
        assignment = zip(self.parents,x)
        pa = np.asarray([q.reduce(assignment,inplace=False).values for q in self.interventional_distributions])
        return pa
        
    
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
        self.parent_assignments = self.generate_binary_assignments()
    
    @classmethod
    def create(cls,N,epsilon):
        q = cls.most_unbalanced_q(N,2)
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
        
    
    def calculate_m(self,qij_sorted):
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
        mq = self.calculate_m(ordered)
        unbalanced = sort_order[0:mq]
        eta[unbalanced] = 1.0/(2*mq)
        return eta,mq

      
           
class ParallelConfounded(Model):
    """ Represents a parallel bandit with one common confounder. Z ->(X1 ... XN) and (X1,...,XN) -> Y 
        Actions are do(x_1 = 0),...,do(x_N = 0), do(x_1=1),...,do(x_N = 1),do(Z=0),do(Z=1),do()"""
    
    def __init__(self,q10,q11,q20,q21,pZ,N1,N2,epsilon):

        pXgivenZ0 = np.hstack((np.full(N1,q10),np.full(N2,q20)))
        pXgivenZ1 = np.hstack((np.full(N1,q11),np.full(N2,q21)))
        self.N1 = N1
        self.N2 = N2
        self.q10,self.q11,self.q20,self.q21 = q10,q11,q20,q21
        self.N = len(pXgivenZ0)
        self.indx = np.arange(self.N)
        self.K = 2*self.N + 3 
        self.pZ = pZ
        self.pX0 = np.vstack((1.0-pXgivenZ0,pXgivenZ0)) # PX0[i,j] = P(X_i = j|Z = 0)
        self.pX1 = np.vstack((1.0-pXgivenZ1,pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        self.pX =  (1-self.pZ)*self.pX0 + self.pZ*self.pX1  # pX[i,j]  = P(X_i = j) = P(Z=0)*P(X_i = j|Z = 0)+P(Z=1)*P(X_i = j|Z = 1)
        self.epsilon = epsilon
        self.epsilon2 = self.pX[1,0]/self.pX[0,0]*self.epsilon
        self.pre_compute()     
    
    def __str__(self):
        string = "ParallelConfounded_mis{0:.1f}_Nis{1}_N1is{2}_qis{3:.1f}_{4:.1f}_{5:.1f}_{6:.1f}_pzis{7:.1f}_epsilonis{8:.1f}".format(self.m,self.N,self.N1,self.q10,self.q11,self.q20,self.q21,self.pZ,self.epsilon)
        return string.replace(".","-")
        
               
    @classmethod
    def create(cls,N,N1,pz,q,epsilon):
        """ builds ParallelConfounded model"""
        q10,q11,q20,q21 = q
        N2 = N - N1
        model = cls(q10,q11,q20,q21,pz,N1,N2,epsilon)
        # adjust costs for do(Z=1), do(Z=0) such that the actions have expected reward .5 to match worse case 
        costs = np.zeros(model.K)
        costs[-2] = model.expected_Y[-2]-.5 
        costs[-3] = model.expected_Y[-3]-.5
        model.set_action_costs(costs)
        return model
        
    def pYgivenX(self,x):
        if x[0] == 1:
            return .5+self.epsilon
        return .5-self.epsilon2
        
    def sample(self,action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """         
        if action == self.K - 2: # do(z = 1)
            x = binomial(1,self.pX1[1,:])
        elif action == self.K - 3: # do(z = 0)
            x = binomial(1,self.pX0[1,:])
        else: # we are not setting z
            x = binomial(1,self.pX[1,:])
            if action != self.K - 1: # not do()
                 i,j = action % self.N, action/self.N
                 x[i] = j
        y = binomial(1,self.pYgivenX(x))
        return x,y
        
              
    def P(self,x):
        """ calculate P(X = x|a) for each action a. 
            x is an array of length N specifiying an assignment to the parents of Y
            returns a vector of length K. 
        """
        # for do(), do(x_i=j)
        p = self.pX[x,self.indx] # vector of lenght K, p[i] = P(X_i = x_i| do())
        joint_j = prod_all_but_j(p) # vector of length K, joint_j = prod_k!=j(X_k = x_k)
        pij = np.vstack((joint_j,joint_j))
        pij[1-x,self.indx] = 0 # 2*N array, pij[i,j] = P(X=x|do(X_i=j)) = d(X_i-j)*prod_k!=j(X_k = x_k)
        pij = pij.reshape((len(x)*2,)) #flatten first N-1 will be px=0,2nd px=1
        p_obs = p.prod()
        
        # for do(z)
        pz0 = self.pX0[x,self.indx].prod() # vector of length K p0[i] = P(X_i = x_i|do(z=0))
        pz1 = self.pX1[x,self.indx].prod() # vector of length K p0[i] = P(X_i = x_i|do(z=1))
        
        result = np.hstack((pij,pz0,pz1,p_obs))
        return result
        
 
    def random_eta_short(self):
        weights = np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
        eta0 = np.random.random(7)
        eta0 = eta0/np.dot(weights,eta0)
        return eta0
        
        
    def weights(self):
        return np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
        
        
    def find_eta(self,tol = 1e-6):
        weights = np.asarray([self.N1,self.N2,self.N1,self.N2,1,1,1])
        eta0 = self.random_eta_short()
        constraints=({'type':'eq','fun':lambda eta: np.dot(eta,weights)-1.0})
        res = minimize(self.m_rep,eta0,bounds = [(0.0,1.0)]*7, constraints = constraints ,options={'disp': True},method='SLSQP',tol=tol)
        assert res.success, " optimisation failed "+res.message
        eta_full = self.expand_eta(res.x)
        return eta_full,res.fun
        

    def m_rep(self,eta_short_form):
        eta = self.expand_eta(eta_short_form)
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def expand_eta(self,eta_short_form):
        eta = np.hstack((
            np.full(self.N1,eta_short_form[0]),
            np.full(self.N2,eta_short_form[1]),
            np.full(self.N1,eta_short_form[2]),
            np.full(self.N2,eta_short_form[3]),
            eta_short_form[4],
            eta_short_form[5],
            eta_short_form[6]
        ))
        return eta  
        
        




















    
