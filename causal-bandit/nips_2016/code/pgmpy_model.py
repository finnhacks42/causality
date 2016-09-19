# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:12:26 2016

@author: finn
"""
from models import Model
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
import numpy as np
from itertools import chain

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