# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:25:51 2016

@author: finn
"""
from pgmpy_model import GeneralModel
from models import ParallelConfoundedNoZAction,ParallelConfounded
from pgmpy.inference import VariableElimination
import numpy as np

#N = 3
#N1 = 1
#pz = .1
#q =.9,.9,.9,.1# .1,.3,.4,.7
#epsilon = .2



N = 5
N1 = 1
pz = .1
q = (.9,.1,.9,.1)
epsilon = .1

model = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon,act_on_z = False)
model2 = ParallelConfoundedNoZAction.create(N,N1,pz,q,epsilon)



for s in xrange(10):
    x,y = model2.sample(model.K-1)
    print x,y
        
#         
#            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
#            self.trials += xij
#            self.success += y*xij
#
#print model.expected_Y
#print model2.expected_Y

#model.expected_Y_observational()



#model1 = ParallelConfounded.create(N,N1,pz,q,epsilon)          
#model2 = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon)
#        
#
#pXgivenZ0 = np.zeros(model2.N)
#pXgivenZ1 = np.zeros(model2.N)
#pX = np.zeros(model2.N)
#for indx,x in enumerate(model2.parents):
#    _,dist = model2.observational_inference.query([x],evidence={'Z':0})
#    _,dist1 = model2.observational_inference.query([x],evidence={'Z':1})
#    _,disto = model2.observational_inference.query([x])
#    pXgivenZ0[indx] = dist.reduce([(x,0)],inplace=False).values
#    pXgivenZ1[indx] = dist1.reduce([(x,0)],inplace=False).values
#    pX[indx] = disto.reduce([(x,0)],inplace=False).values
#  
#
#
#print "PXgivenZ0"      
#print pXgivenZ0
#print model1.pX0[0]
#print "-----------"
#
#print "pXgivenZ1"
#print pXgivenZ1
#print model1.pX1[0]
#print "----------"



   
#x = np.zeros(N,dtype=int)
#
#print model1.P(x)
#print model2.P(x)
#
#dox10 = model2.post_action_models[0]
#infer = VariableElimination(dox10)
#marginals,joint = infer.query(model2.parents)
#print joint
#for v in model2.parents:
#    print marginals[v]
#        
    