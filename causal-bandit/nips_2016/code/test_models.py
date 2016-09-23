# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:11:22 2016

@author: finn
"""

import unittest
import numpy.testing as np_test
import numpy as np
from models import ParallelConfounded,VeryConfounded
from pgmpy_model import GeneralModel
from pgmpy.inference import VariableElimination


#TODO add test to ensure parrallel confounded has desired reward structure

class TestModelVeryConfoundedEquivalence(unittest.TestCase):
    def setUp(self):
        Nz = 3
        pZ1 = .1
        pZ = .4
        a = .3
        b = .7
        py = np.asarray([.1,.5,.3,.2])
    
        self.model1 = GeneralModel.create_very_confounded(Nz,pZ1,pZ,a,b,py)
        self.model2 = VeryConfounded(a,b,Nz,pZ1,pZ,py)
        
        px1givenA = []
        px2givenA = []
        for m in self.model1.post_action_models:
            infer = VariableElimination(m)
            marginals,_ = infer.query(['X1','X2'])
            px1givenA.append(marginals['X1'].reduce([('X1',1)],inplace=False).values)
            px2givenA.append(marginals['X2'].reduce([('X2',1)],inplace=False).values)        
        
        self.px1givenA = np.asarray(px1givenA)
        self.px2givenA = np.asarray(px2givenA)
        
             
    def test_px1givenA(self):
        np_test.assert_array_almost_equal(self.px1givenA,self.model2.pX1givenA[1])
        
    def test_px2givenA(self):
        np_test.assert_array_almost_equal(self.px2givenA,self.model2.pX2givenA[1])
        
    def test_P_simulate(self): #TODO
        
    def test_P(self):
        #for x in self.model1.generate_binary_assignments():
        x = np.asarray([0,0])
        np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),decimal=3)
        
#    def test_rewards(self):
#        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards,decimal=2)
            
        
#class TestModelParallelConfoundedEquivalence(unittest.TestCase):
#    
#    def setUp(self):
#        N = 7
#        N1 = 2
#        pz = .1
#        q = .1,.3,.4,.7
#        epsilon = .2
#        self.model1 = ParallelConfounded.create(N,N1,pz,q,epsilon)          
#        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon)
#        
#    def test_P(self):
#        assignments = self.model1.generate_binary_assignments()
#        for x in assignments:
#            np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),decimal=2)
#           
#            
#    def test_rewards(self):
#        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards)
            
        
