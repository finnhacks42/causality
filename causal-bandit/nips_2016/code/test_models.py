# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:11:22 2016

@author: finn
"""

import unittest
import numpy.testing as np_test
import numpy as np
from models import ParallelConfounded,ParallelConfoundedNoZAction,Parallel,ScaleableParallelConfounded,ScaleableParallelConfoundedNoZAction
from pgmpy_model import GeneralModel
from itertools import chain,product




class TestSamplingAndExpectedRewardsMatch(unittest.TestCase):
    
    def setUp(self):
        self.N = 3
        self.pz = .1
        self.N1 = 1
        self.q = (.1,.2,.3,.4)
        self.pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.epsilon = .2      
        return
        
    def estimate_px_and_y_from_samples(self,model,samples):
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
        
    def assert_samples_consistent_probabilities(self,model,samples):
        xcounts,samples_y = self.estimate_px_and_y_from_samples(model,samples)
        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
        for x in model.get_parent_assignments():
            p_in_sample = [xcounts[tuple(chain([a],x))] for a in range(model.K)]
            np_test.assert_almost_equal(model.P(x),p_in_sample,decimal=2)
        
        
    def test_parallel_confounded(self):
        model = ParallelConfounded.create(self.N,self.N1,self.pz,self.pY,self.q)
        self.assert_samples_consistent_probabilities(model,50000)
        
    def test_scalable_confounded(self):
        model = ScaleableParallelConfoundedNoZAction(self.q,self.pz,self.pY,self.N1,self.N-self.N1)
        self.assert_samples_consistent_probabilities(model,50000)
        
    def test_scalable_noz(self):
        model = ScaleableParallelConfoundedNoZAction(self.q,self.pz,self.pY,self.N1,self.N-self.N1)
        self.assert_samples_consistent_probabilities(model,50000)
        
    
            
    

            
            
        
        
#    def test_parallel_confounded_no_z(self):
#        model = ParallelConfoundedNoZAction.create(self.N,self.m,self.pz,self.q,self.epsilon)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=3)
#              
#    def test_parallel(self):
#        model = Parallel.create(self.N,self.m,self.epsilon)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
#        
#    def test_general_confounded_no_z(self):
#        model = GeneralModel.create_confounded_parallel(self.N,self.m,self.pz,self.q,self.epsilon,act_on_z=False)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
#        
#    def test_general_confounded(self):
#        model = GeneralModel.create_confounded_parallel(self.N,self.m,self.pz,self.q,self.epsilon,act_on_z=True)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
#        
        
        
        
    


class TestModelParallelConfoundedNoZEquivalence(unittest.TestCase):
    
    def setUp(self):
        N = 3
        N1 = 1
        pz = .1
        q = 1,1,1,0
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.model1 = ParallelConfoundedNoZAction.create(N,N1,pz,pY,q)          
        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,pY,q,act_on_z=False)
        self.model3 = ScaleableParallelConfoundedNoZAction(q,pz,pY,N1,N-N1)
        
    def test_P(self):
        assignments = self.model1.get_parent_assignments()
        for x in assignments:
            p1 = self.model1.P(x)
            p2 = self.model2.P(x)
            p3 = self.model3.P(x)
            np_test.assert_array_almost_equal(p1,p2,decimal=2,err_msg="x:"+str(x))
            np_test.assert_array_almost_equal(p2,p3,decimal=2,err_msg="x:"+str(x))
           
           
    def test_rewards(self):
        r1 = self.model1.expected_rewards
        r2 = self.model2.expected_rewards
        r3 = self.model3.expected_rewards
        np_test.assert_array_almost_equal(r1,r2)
        np_test.assert_array_almost_equal(r2,r3)
        
        
    def test_expected_y(self):
        y1 = self.model1.expected_Y
        y2 = self.model2.expected_Y
        y3 = self.model3.expected_Y
        np_test.assert_array_almost_equal(y1,y2)
        np_test.assert_array_almost_equal(y2,y3)
        
    def test_m(self):
        difference = abs(self.model1.m - self.model2.m)
        self.assertLess(difference,.5)
        
        
    def test_V(self):
        for s in xrange(100):
            eta_short = self.model3.random_eta_short()
            eta = self.model3.expand(eta_short)
            
            v1 = self.model1.V(eta)
            v2 = self.model2.V(eta)
            v3 = self.model3.V(eta)
            np_test.assert_array_almost_equal(v1,v2,err_msg="v1 != v2, eta:"+str(eta_short))
            np_test.assert_array_almost_equal(v2,v3,err_msg="v2 != v3, eta:"+str(eta_short))


           
class TestScaleableModelEquivalance(unittest.TestCase):
    def setUp(self):
        N = 5
        N1 = 2
        q = .1,.3,.4,.7
        pZ = .2
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.model1 = ParallelConfounded.create(N,N1,pZ,pY,q)
        self.model2 = ScaleableParallelConfounded(q,pZ,pY,N1,N-N1)
        self.N1 = N1
        self.N2 = N - N1
        
    def test_P(self):
        for x in self.model1.get_parent_assignments():
            np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),err_msg="x:"+str(x))

    def test_V(self):
        for t in range(10):
            eta_short = self.model2.random_eta_short()
            eta = self.model2.expand(eta_short)
            
            v1 = self.model1.V(eta)
            v2 = self.model2.expand(self.model2.V_short(eta_short))
            
            np_test.assert_array_almost_equal(v1,v2,err_msg="eta:"+str(eta))
    
    def test_rewards(self):
        np_test.assert_almost_equal(self.model1.expected_Y,self.model2.expected_Y)
        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards)
        
        
class TestModelParallelConfoundedEquivalence(unittest.TestCase):
    
    def setUp(self):
        N = 5
        N1 = 2
        pz = .2
        q = .1,.3,.4,.7
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.model1 = ParallelConfounded.create(N,N1,pz,pY,q)          
        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,pY,q)
        
        
    def test_pxgivenz(self):
        pXgivenZ0 = np.zeros(self.model2.N)
        pXgivenZ1 = np.zeros(self.model2.N)
        for indx,x in enumerate(self.model2.parents):
            _,dist = self.model2.observational_inference.query([x],evidence={'Z':0})
            _,dist1 = self.model2.observational_inference.query([x],evidence={'Z':1})
            pXgivenZ0[indx] = dist.reduce([(x,1)],inplace=False).values
            pXgivenZ1[indx] = dist1.reduce([(x,1)],inplace=False).values
        
        np_test.assert_almost_equal(pXgivenZ0,self.model1.pXgivenZ[1,:,0])
        np_test.assert_almost_equal(pXgivenZ1,self.model1.pXgivenZ[1,:,1])
   
    def test_V(self):
        for t in range(10):
            eta = self.model1.random_eta()
            v1 = self.model1.V(eta)
            v2 = self.model2.V(eta)
            np_test.assert_array_almost_equal(v1,v2,err_msg="eta:"+str(eta))
        
        
    def test_P(self):
        assignments = self.model1.get_parent_assignments()
        for x in assignments:
            np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),err_msg="x:"+str(x))
            
    def test_expected_y(self):
        np_test.assert_almost_equal(self.model1.expected_Y,self.model2.expected_Y)
           
            
    def test_rewards(self):
        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards)
        
    def test_m(self):
        difference = abs(self.model1.m - self.model2.m)
        self.assertLess(difference,.5)
            
        
