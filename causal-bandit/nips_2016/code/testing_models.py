# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:25:51 2016

@author: finn
"""
from pgmpy_model import GeneralModel
from models import ScaleableParallelConfoundedNoZAction, ParallelConfoundedNoZAction
from pgmpy.inference import VariableElimination
import numpy as np
from algorithms import ParallelCausal,ObservationalEstimate,UniformSampling
from scipy.optimize import minimize

#N = 3
#N1 = 1
#pz = .1
#q =.9,.9,.9,.1# .1,.3,.4,.7
#epsilon = .2

np.set_printoptions(precision=2)

def argmax_2(array):
    best = np.argmax(array)
    a_copy = array.copy()
    a_copy[best] = -1
    next_best = np.argmax(a_copy)
    return (best,next_best)
    
def argmax_excluding(array,indx):
    a = array.copy()
    a[indx] = -1
    return np.argmax(a)


def epsilon_gap(expected,observed):
    best = np.argmax(expected)
    next_best = argmax_excluding(expected,best)
    epsilon1 = expected[best] - expected[next_best]
    epsilon2 = observed[best] - observed[argmax_excluding(observed,best)]
    return epsilon1,epsilon2
    
def epsilon_gap2(expected,observed):
    ex = np.around(expected,2) # round to 2 decimal places as epsilon < .01 negligable for T ~ 400
    observed = np.around(observed,2)    
    ex_vals = np.unique(ex)
    if len(ex_vals) == 1:
        epsilon1 = 0
    else:
        epsilon1 = ex_vals[-1] - ex_vals[-2]
        
    ex_best_indx = np.where(np.isclose(ex,ex_vals[-1]))[0]
    
    best_obs = np.argmax(observed)
    if best_obs in ex_best_indx:
        bias = 0
    
    else:
        bias =   ex[best_obs]- ex_vals[-1]
    
    #epsilon2 = max(observed[ex_best_indx]) - observed[argmax_excluding(observed,ex_best_indx)]
    
    
    return epsilon1,bias
    
    
def rewards(pz,q,pY):
    q10,q11,q20,q21 = q
    a,b,c,d = pY
    
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
    
    expected_rewards = np.asarray([dox10,doxj,dox20,dox11,doxj,dox21,doxj])
    observational_rewards = np.asarray([x10,xj0,x20,x11,xj1,x21,doxj])
    
    return (expected_rewards,observational_rewards)
    

def random_minimize(func,input_func,iters):
    min_value = None
    best_input = None
    for i in xrange(iters):
        input_value = input_func()
        output = func(input_value)
        if min_value is None or output < min_value:
            min_value = output
            best_input = input_value
    return min_value,best_input
    
    

#pz = .5
q = (.2,.8,.7,.3)

def func(pYandpz):
    pY = pYandpz[0:4]
    pz = pYandpz[-1]
    expected,observed = rewards(pz,q,pY)
    ep1,ep2 = epsilon_gap2(expected,observed)
    return ep1*ep2 #-ep1/ep2
    
value,x = random_minimize(func,lambda: np.random.random(5),50000)
py = x[0:4]
pz = x[-1]
ex,obs = rewards(pz,q,py)
print pz,py
print ex
print obs
print epsilon_gap2(ex,obs)

pY = np.reshape(py,(2,2))


N0 = 5
N1 = 1
N2 = 2
N = N0+N1+N2
q10,q11,q20,q21 = q
pXgivenZ0 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q10),np.full(N2,q20)))    #(np.full(N1,q10),np.full(N2,q20))
pXgivenZ1 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q11),np.full(N2,q21)))
pXgivenZ = np.stack((np.vstack((1.0-pXgivenZ0,pXgivenZ0)),np.vstack((1.0-pXgivenZ1,pXgivenZ1))),axis=2) # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
pYfunc = lambda x: pY[x[N0],x[N-1]]
model = ParallelConfoundedNoZAction(pz,pXgivenZ,pYfunc)

alg = ObservationalEstimate()
alg2 = ParallelCausal()
alg3 = UniformSampling()
r = alg.run(10000,model)
r2 = alg2.run(10000,model)
r3 = alg3.run(10000,model)

print ex
print model.expected_Y
print alg3.u
print "\n"



print obs
print alg.u
print alg2.u




#0.138298168853 [ 0.46  0.41  0.01  0.95]
#0.409695172076 [ 0.09  0.79  0.97  0.04] (.12,-.12)

#for i in xrange(500):
#    pY0 = np.random.random(5)
#    res = minimize(func,pY0,bounds = [(0.1,.9)]*5,options={'disp': False, 'maxiter':200})
#    if res.success:
#        py = res.x[0:4]
#        pz = res.x[-1]
#        print pz,py
#        print res.fun
#        ex,obs = rewards(pz,q,py)
#        print ex
#        print obs
#        print epsilon_gap(ex,obs)
#        print "\n"


#[ 0.14  0.33  0.48  0.16]
   
#pz = .5
##q = (.1,.99,.2,.7)
#q = (.1,.9,.7,.3)
#q10,q11,q20,q21 = q
##a,b,c,d = .4,.3,.7,.6
#a,b,c,d = .6,.1,0,1
#
#
#
#
#print epsilon_gap(expected_rewards,observational_rewards)
#
#
#
#costs = expected_rewards - 0.50
#costs[1] -= .1
#
#print expected_rewards
#print observational_rewards
#print expected_rewards - observational_rewards
#print "\n"
#print costs
#print expected_rewards - costs
#print observational_rewards - costs
#
#
#N = 10
#N1 = 1


#print dox10,x10,x10-dox10
#print dox11,x11,x11-dox11
#print dox20,x20,x20-dox20
#print dox21,x21,x21-dox21
#print doxj,xj0, xj0 - doxj
#print doxj,xj1, xj1 - doxj

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
    