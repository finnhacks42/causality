# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from models import ParallelConfounded,ScaleableParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,AlphaUCB,ThompsonSampling
from experiment_config import Experiment
import numpy as np


def regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = 1000): 
    m_vals = []
    models = []
    regret = np.zeros((len(algorithms),len(N1_vals),simulations))

    for m_indx,N1 in enumerate(N1_vals):
        model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1,compute_m = False)
        eta = [0,0,1.0/(N1+2.0),0,0,0,1-N1/(N1+2.0)]
        model.compute_m(eta_short = eta)
        # find the v 
        
        print N1,model.m
        m_vals.append(model.m)
        models.append(model)
        for a_indx, algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
                
    
    return m_vals,regret,models
    
   
experiment = Experiment(4)
experiment.log_code()
    
N = 50
N1_vals = range(1,N,3)
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .3
simulations = 10000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal(),AlphaUCB(2),ThompsonSampling()]
#pY = np.asarray([[.4,.4],[.7,.7]])

epsilon = .3
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

m_vals,regret,models = regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = simulations)

experiment.plot_regret(regret,m_vals,"m",algorithms,legend_loc = "lower right")
experiment.log_state(globals())


#etas = np.zeros((len(m_vals),7))
#for m_indx,model in enumerate(models):
#    etas[m_indx] = model.eta_short

#import matplotlib.pyplot as plt
#
#fig,ax = plt.subplots()
#for m_indx in range(len(models)):
#    eta = etas[m_indx,:]
#    print len(eta)
#    print len(m_vals)
#    ax.plot(m_vals,eta)

#        for indx,alg in enumerate(algorithms):    
#            ax.errorbar(xvals,mu[indx,:],yerr = error[indx,:],label = alg.label,linestyle="",marker = alg.marker,color=alg.color)
#        ax.set_xlabel(xlabel)
#        ax.set_ylabel(self.REGRET_LABEL)

# compare against N1  =N1_vals[7]
#N1 = N1_vals[6]
#N1b = N1_vals[7]

#alg = GeneralCausal(truncate = "none")
#model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)
#
#optimals = [0,1,N-1,N,N+1,2*N-1,2*N,2*N+1,2*N+2]
#opt = [model.action_tuple(i) for i in optimals]
#
#regret = np.zeros((len(optimals),simulations))
#for indx,i in enumerate(optimals):
#    model.make_ith_arm_epsilon_best(epsilon,i)
#    print i
#    for s in xrange(simulations):
#        regret[indx,s] = alg.run(T,model)
#        
#mu = np.zeros((model.K,simulations))
#for s in xrange(simulations):
#    r = alg.run(T,model)
#    mu[:,s] = alg.u
    

#diffs = np.zeros(model.K)
#for s in xrange(simulations):
#    regret = alg.run(T,model)
#    diffs += (alg.u - model.expected_Y)
   
    


#
#models = []
#for trial in range(5):
#    model = ScaleableParallelConfounded(q,pz,pY,N1a,N-N1a)
#    model.compute_m()
#    i = np.argmax(model.V(model.eta))
#    model.make_ith_arm_epsilon_best(epsilon,i)
#    models.append(model)
#
#
#
#

#
#regret = np.zeros((len(models),simulations))
#for s in range(simulations):
#    for i,model in enumerate(models):
#        regret[i,s] = alg.run(T,model)


#model2 = ScaleableParallelConfounded(q,pz,pY,N1b,N-N1b)
#model2.compute_m()
#i2 = np.argmax(model2.V(model2.eta))
#model2.make_ith_arm_epsilon_best(epsilon,i2)







