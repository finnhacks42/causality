# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:25:51 2016

@author: finn
"""
from pgmpy_model import GeneralModel
from models import ParallelConfoundedNoZAction,ParallelConfounded
from pgmpy.inference import VariableElimination
import numpy as np
from algorithms import ParallelCausal

#N = 3
#N1 = 1
#pz = .1
#q =.9,.9,.9,.1# .1,.3,.4,.7
#epsilon = .2



N = 10
N1 = 1
pz = .4
q = (.1,.9,.2,.7)
q10,q11,q20,q21 = q
a,b,c,d = .4,.4,.7,.7
epsilon = .1
T=2000

# P(X1,X2)
alpha = (1-pz)*(1-q10)*(1-q20)+pz*(1-q11)*(1-q21)
beta = (1-pz)*(1-q10)*q20+pz*(1-q11)*q21
gamma = (1-pz)*q10*(1-q20)+pz*q11*(1-q21)
delta = (1-pz)*q10*q20+pz*q11*q21

#P(Z,X2)
a1 = (1-pz)*(1-q20)
b1 = (1-pz)*q20
c1 = pz*(1-q21)
d1 = pz*q21

#P(Z|X2=0)
z00 = a1/(a1+c1)
z01 = c1/(a1+c1)

#P(Z|X2=1)
z10 = b1/(b1+d1)
z11 = d1/(b1+d1)

dox10 = a*((1-pz)*(1-q20)+pz*(1-q21)) + b*((1-pz)*q20+pz*q21)
x10 = a*(alpha/(alpha+beta))+b*(beta/(alpha+beta))

dox11 = c*((1-pz)*(1-q20)+pz*(1-q21)) + d*((1-pz)*q20+pz*q21)
x11 = c*(gamma/(gamma+delta))+d*(delta/(gamma+delta))

dox20 = a*((1-pz)*(1-q10)+pz*(1-q11))+c*((1-pz)*q10+pz*q11)
x20 = a*(alpha/(alpha+gamma))+c*(gamma/(alpha+gamma))

dox21 = b*((1-pz)*(1-q10)+pz*(1-q11))+d*((1-pz)*q10+pz*q11)
x21 = b*(beta/(beta+delta))+d*(delta/(beta+delta))

doxj = a*alpha+b*beta+c*gamma+d*delta
xj0 = a*(z00*(1-q10)*(1-q20)+z01*(1-q11)*(1-q21))+b*(z00*(1-q10)*q20+z01*(1-q11)*q21)+c*(z00*q10*(1-q20)+z01*q11*(1-q21))+d*(z00*q10*q20+z01*q11*q21) 
xj1 = a*(z10*(1-q10)*(1-q20)+z11*(1-q11)*(1-q21))+b*(z10*(1-q10)*q20+z11*(1-q11)*q21)+c*(z10*q10*(1-q20)+z11*q11*(1-q21))+d*(z10*q10*q20+z11*q11*q21) 

print dox10,x10,x10-dox10
print dox11,x11,x11-dox11
print dox20,x20,x20-dox20
print dox21,x21,x21-dox21
print doxj,xj0, xj0 - doxj
print doxj,xj1, xj1 - doxj

#alg = ParallelCausal()
#model = ParallelConfoundedNoZAction.create(N,N1,pz,q,epsilon)
#model.make_ith_arm_epsilon_best(epsilon,0)


#epsilon = .1
#
##model = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon,act_on_z = False)
#model2 = ParallelConfoundedNoZAction.create(N,N1,pz,q,epsilon)
#
#trials = np.zeros(model2.K)
#success = np.zeros(model2.K)
#
#for s in xrange(1000):
#    x,y = model2.sample(model2.K-1)
#    xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
#    trials += xij
#    success += y*xij
#
#mu = np.true_divide(success,trials)
#r = model2.expected_Y
#
#
#print r[0],mu[0]
#print r[N],mu[N]
#print r[N-1],mu[N-1]
#print r[2*N-1],mu[2*N-1]
#print r[1],mu[1],mu[N+1]




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
#dox10 = mode,mul2.post_action_models[0]
#infer = VariableElimination(dox10)
#marginals,joint = infer.query(model2.parents)
#print joint
#for v in model2.parents:
#    print marginals[v]
#        
    