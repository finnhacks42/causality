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
from itertools import chain,product
from scipy.special import expit

class GeneralModel(Model):
    """ Allows construction of an arbitray causal graph & action space with discrete (currently assumed binary) CPD tables. 
        This implementation will not scale to large graphs. """
    def __init__(self,model,actions,py_func):
        """ model is a pgmpy.BayesianModel
            actions is a list of (var,value) tuples """
        self.py_func = py_func
        self.parents = sorted(model.get_parents('Y'))
        self.N = len(self.parents)
        self.actions = actions
        self.K = len(actions)
        
        self.observational_model = model
        self.observational_inference = VariableElimination(self.observational_model)
        self.post_action_models = [GeneralModel.do(model,action) for action in actions]
        self.samplers = [BayesianModelSampling(model_a) for model_a in self.post_action_models]
        
        self.interventional_distributions = []
        for indx,new_model in enumerate(self.post_action_models):
            infer = VariableElimination(new_model)
            _,distribution_over_parents = infer.query(self.parents)
            self.interventional_distributions.append(distribution_over_parents)        
           
        self.pre_compute()
        
    def expected_Y_observational(self):
        """ return a vector of length K with the expected Y given we observe the variable-value pair corresponding to each action """
        expected_Y = np.zeros(self.K)
        
        for indx,action in enumerate(self.actions):
            var,value = action
            if var is None:
                _,distribution = self.observational_inference.query(['Y'])
            else:
                _,distribution = self.observational_inference.query(['Y'],evidence = dict([action]))
            
            pyis1 = distribution.reduce([('Y',1)],inplace=False).values
            expected_Y[indx] = pyis1
            
        return expected_Y
        
    def _expected_Y(self):
        expected_Y = np.zeros(self.K)
        for indx,new_model in enumerate(self.post_action_models):
            infer = VariableElimination(new_model)
            _,distribution_over_reward = infer.query(['Y'])
            expected_reward = distribution_over_reward.reduce([('Y',1)],inplace=False).values #TODO investigate failing if inplace=True - bug in pgmpy?
            expected_Y[indx] = expected_reward
        return expected_Y
        
    @staticmethod
    def build_ycpd(py_func,N):
        cpd = np.zeros((2,2**N))
        for i,x in enumerate(Model.generate_binary_assignments(N)):
            cpd[0,i] = 1 - py_func(x)
            cpd[1,i] = py_func(x)
        return cpd
        
    def pYgivenX(self,x):
        return self.py_func(x)
        
        
    @classmethod
    
    def create_confounded_parallel(cls,N,N1,pz,pY,q,epsilon, act_on_z = True):       
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

           
        def py(x):
             i,j = x[0],x[N-1]
             return pY[i,j] 
        
        
        ycpd = GeneralModel.build_ycpd(py,N)
        cpds.append(TabularCPD(variable='Y',variable_card=2, values = ycpd, evidence = xvars,evidence_card = [2]*len(xvars)))
        
        model.add_cpds(*cpds)
        model.check_model()
        
        if act_on_z:
            actions = list(chain([(x,0) for x in xvars],[(x,1) for x in xvars],[("Z",i) for i in (0,1)],[(None,None)]))
              
        else:
            actions = list(chain([(x,0) for x in xvars],[(x,1) for x in xvars],[(None,None)]))
            
        pgm_model = cls(model,actions,py)  
       
        return pgm_model
    
    @classmethod    
    def create_very_confounded(cls,Nz,pZ1,pZ,a,b,py):
        """ construct a very confounded model """
    
        zvars = ['Z'+str(i) for i in range(1,Nz+1)]
        xvars = ['X'+str(i) for i in range(1,3)]
        edges = chain(product(zvars,xvars),product(xvars,['Y']))
        bayes_model = BayesianModel(edges)
        
        z_other = list(product((0,1),repeat=(Nz-1)))
        
        px1 = np.hstack((np.full(2**(Nz-1),a),[np.mean(z) for z in z_other]))       
        px2 = np.hstack((np.full(2**(Nz-1),b),[np.prod(z) for z in z_other]))
        
        cpds = [TabularCPD(variable = 'Z1',variable_card=2,values = np.vstack((1-pZ1,pZ1)) )]
        cpds.extend([TabularCPD(variable = v,variable_card=2,values = np.vstack((1-pZ,pZ)) ) for v in zvars[1:]])
        cpds.append(TabularCPD(variable = 'X1', variable_card = 2, values = np.vstack((1-px1,px1)),evidence=zvars,evidence_card = [2]*Nz))
        cpds.append(TabularCPD(variable = 'X2', variable_card = 2, values = np.vstack((1-px2,px2)),evidence=zvars,evidence_card = [2]*Nz))
        cpds.append(TabularCPD(variable = 'Y', variable_card = 2, values = np.vstack((1-py,py)),evidence=xvars,evidence_card = [2]*len(xvars)))
        
        bayes_model.add_cpds(*cpds)
        bayes_model.check_model()
        actions = list(chain([(z,0) for z in zvars],
                             [(z,1) for z in zvars],
                             [(x,0) for x in xvars],
                             [(x,1) for x in xvars],
                             [(None,None)]))
        
        model = cls(bayes_model,actions)
        return model
    
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
        
if __name__ == "__main__":  
    N = 8
    N1 = 1
    pz = .1
    q = (.9,.1,.9,.1)
    epsilon = .1

    model = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon,act_on_z = False)
    print model.expected_Y - model.expected_Y_observational()
    
#    
#    pXgivenZ0 = np.zeros(N)
#    pXgivenZ1 = np.zeros(N)
#    for indx,x in enumerate(model.parents):
#        _,dist = model.observational_inference.query([x],evidence={'Z':0})
#        _,dist1 = model.observational_inference.query([x],evidence={'Z':1})
#        pXgivenZ0[indx] = dist.reduce([(x,1)],inplace=False).values
#        pXgivenZ1[indx] = dist1.reduce([(x,1)],inplace=False).values
   
    
    

    
    
    

