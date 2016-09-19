# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:11:22 2016

@author: finn
"""

import unittest
import numpy.testing as np_test
from models import ParallelConfounded,GeneralModel

#TODO add test to ensure parrallel confounded has desired reward structure

class TestModelParallelConfoundedEquivalence(unittest.TestCase):
    
    def setUp(self):
        N = 7
        N1 = 2
        pz = .1
        q = .1,.3,.4,.7
        epsilon = .2
        self.model1 = ParallelConfounded.create(N,N1,pz,q,epsilon)          
        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon)
        
    def test_P(self):
        assignments = self.model1.generate_binary_assignments()
        for x in assignments:
            diff = self.model1.P(x) - self.model2.P(x)
            self.assertLess(max(diff),.01)
            #TODO understand and replace with numpy array almost equal test method
            
    def test_rewards(self):
        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards)
            
        
