# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:11:22 2016

@author: finn
"""

import unittest
import numpy.testing as np_test
import numpy as np
from models import ParallelConfounded,ParallelConfoundedNoZAction,Parallel,ScaleableParallelConfounded
from pgmpy_model import GeneralModel
from itertools import chain,product




#class TestSamplingAndExpectedRewardsMatch(unittest.TestCase):
#    
#    def setUp(self):
#        self.samples = 10000
#        self.N = 3
#        self.pz = .1
#        self.m = 1
#        self.q = (.1,.2,.3,.4)
#        self.pY = np.asanyarray([[.2,.8],[.3,.9]])
#        self.epsilon = .2      
#        return
#        
#    def estimate_px_and_y_from_samples(self,model):
#        expected_y = np.zeros(model.K,dtype=float)
#        shape = list(chain([model.K],[2]*self.N))
#        xcounts = np.zeros(shape)
#        for a in range(model.K):
#            for s in xrange(self.samples):
#                x,y = model.sample(a)
#                pos = tuple(chain([a],x))
#                
#                xcounts[pos] +=1
#               
#                expected_y[a] += y
#            expected_y[a] = expected_y[a]/self.samples
#        xcounts = xcounts/self.samples
#        return xcounts,expected_y
#    
#    def test_parallel_confounded(self):
#        model = ParallelConfounded.create(self.N,self.m,self.pz,self.pY,self.q,self.epsilon)
#        xcounts,samples_y = self.estimate_px_and_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
        
        #for x in model.get_parent_assignments():
        #    np_test.assert_almost_equal(xcounts[:,tuple(x)],model.P(x))
            
            
        
        
#    def test_parallel_confounded_no_z(self):
#        model = ParallelConfoundedNoZAction.create(self.N,self.m,self.pz,self.q,self.epsilon)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=3)
              
#    def test_parallel(self):
#        model = Parallel.create(self.N,self.m,self.epsilon)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
        
#    def test_general_confounded_no_z(self):
#        model = GeneralModel.create_confounded_parallel(self.N,self.m,self.pz,self.q,self.epsilon,act_on_z=False)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
#        
#    def test_general_confounded(self):
#        model = GeneralModel.create_confounded_parallel(self.N,self.m,self.pz,self.q,self.epsilon,act_on_z=True)
#        samples_y = self.estimate_y_from_samples(model)
#        np_test.assert_almost_equal(model.expected_rewards,samples_y,decimal=2)
        
        
        
        
    


#class TestModelParallelConfoundedNoZEquivalence(unittest.TestCase):
#    
#    def setUp(self):
#        N = 3
#        N1 = 1
#        pz = .1
#        q = 1,1,1,0
#        pY = np.asanyarray([[.2,.8],[.3,.9]])
#        epsilon = .2
#        self.model1 = ParallelConfoundedNoZAction.create(N,N1,pz,pY,q,epsilon)          
#        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,pY,q,epsilon,act_on_z=False)
#        
#    def test_P(self):
#        assignments = self.model1.get_parent_assignments()
#        for x in assignments:
#            np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),decimal=2,err_msg="x:"+str(x))
#           
#    def test_rewards(self):
#        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards)
#        
#    def test_expected_y(self):
#        np_test.assert_almost_equal(self.model1.expected_Y,self.model2.expected_Y)
#        
#    def test_m(self):
#        difference = abs(self.model1.m - self.model2.m)
#        self.assertLess(difference,.5)


class TestScaleableModel(unittest.TestCase):
    def setUp(self):
        q = .1,.3,.4,.7
        pZ = .2
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.model = ScaleableParallelConfounded(q,pZ,pY,1,2)

        
    def test_p_x_givenz0(self):
        np_test.assert_array_almost_equal([.6**2,.6*.9,.6*.9,0,0,0,.6*.6*.9,0,.6*.6*.9],self.model.P0(np.asarray([0,0,0],dtype=int))) 
        np_test.assert_array_almost_equal([.6*.4,.9*.4,0,0,0,.9*.6,.9*.6*.4,0,.9*.6*.4],self.model.P0(np.asarray([0,0,1],dtype=int)))
        np_test.assert_array_almost_equal([.4*.6,0,.9*.4,0,.9*.6,0,.9*.4*.6,0,.9*.4*.6],self.model.P0(np.asarray([0,1,0],dtype=int)))
        np_test.assert_array_almost_equal([.4*.4,0,0,0,.9*.4,.9*.4,.9*.4*.4,0,.9*.4*.4],self.model.P0(np.asarray([0,1,1],dtype=int)))
        
    
    
class TestScaleableModelEquivalance(unittest.TestCase):
    def setUp(self):
        N = 3
        N1 = 1
        q = .1,.3,.4,.7
        pZ = .2
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        self.model1 = ParallelConfounded.create(N,N1,pZ,pY,q,.1)
        self.model2 = ScaleableParallelConfounded(q,pZ,pY,N1,N-N1)
        self.N1 = N1
        self.N2 = N - N1
        
#    def test_P(self):
#        for x in self.model1.get_parent_assignments():
#            np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),err_msg="x:"+str(x))
#            
#    def test_P_counts(self):
#        d = {}
#        for x in self.model1.get_parent_assignments():
#            n1,n2 = x[0:self.N1].sum(),x[self.N1:].sum()
#            p1 = self.model1.P(x)
#            v = d.get((n1,n2),0)
#            d[(n1,n2)]=v+p1
#        for key,value in d.iteritems():
#            np_test.assert_array_almost_equal(value,self.model2.expand(self.model2.p_of_count_given_action(*key)))
#            
#            
#    def test_P_counts_sum(self):
#        result = np.zeros(7)
#        for n1,n2 in product(range(self.model2.N1+1),range(self.model2.N2+1)):
#            result+=self.model2.p_of_count_given_action(n1,n2)
#        np_test.assert_array_almost_equal(np.ones(7),result)
            
            
    def test_V(self):
        for t in range(10):
            eta = self.model1.random_eta_short()
            v1 = self.model1.V(self.model1.expand_eta(eta))
            v2 = self.model2.expand(self.model2.V_short(eta))
            np_test.assert_array_almost_equal(v1,v2,err_msg="eta:"+str(eta))
        
        
class TestModelParallelConfoundedEquivalence(unittest.TestCase):
    
    def setUp(self):
        N = 5
        N1 = 2
        pz = .2
        q = .1,.3,.4,.7
        pY = np.asanyarray([[.2,.8],[.3,.9]])
        epsilon = .2
        self.model1 = ParallelConfounded.create(N,N1,pz,pY,q,epsilon)          
        self.model2 = GeneralModel.create_confounded_parallel(N,N1,pz,pY,q,epsilon)
        
        
    def test_pxgivenz(self):
        pXgivenZ0 = np.zeros(self.model2.N)
        pXgivenZ1 = np.zeros(self.model2.N)
        for indx,x in enumerate(self.model2.parents):
            _,dist = self.model2.observational_inference.query([x],evidence={'Z':0})
            _,dist1 = self.model2.observational_inference.query([x],evidence={'Z':1})
            pXgivenZ0[indx] = dist.reduce([(x,1)],inplace=False).values
            pXgivenZ1[indx] = dist1.reduce([(x,1)],inplace=False).values
        
        np_test.assert_almost_equal(pXgivenZ0,self.model1.pX0[1])
        np_test.assert_almost_equal(pXgivenZ1,self.model1.pX1[1])
   
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
            
        
