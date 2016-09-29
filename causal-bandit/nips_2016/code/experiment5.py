# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:44:51 2016

Compare the performance of the algorithms on a Confounded Parallel bandit for which 
there is no action to set Z. The Parallel algorithm is mis-specified in this setting, as it assumes there are no confounders.
If the resulting bias exceeds epsilon then the Parallel algorithm will never identify the best arm.
"""


from models import ParallelConfoundedNoZAction
from algorithms import SuccessiveRejects,GeneralCausal,ParallelCausal,RandomArm
from experiment_config import now_string,Experiment
import numpy as np



def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    pulls = np.zeros((len(algorithms),len(T_vals),model.K),dtype=int)
    for T_indx,T in enumerate(T_vals): 
        
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                i = np.random.randint(0,model.K)
                #model.make_ith_arm_epsilon_best(epsilon,0)
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
                if algorithm.best_action is not None:
                    pulls[a_indx,T_indx,algorithm.best_action] +=1
        print T
                
    return regret,pulls
           
simulations = 100

# why doesn't pulls and eta line up
                  
N =20
N1 = 1
pY = np.asarray([[.4,.4],[.7,.7]])
pz = .3
q = (.1,.9,.1,.7)
epsilon=.1
model = ParallelConfoundedNoZAction.create(N,N1,pz,pY,q,epsilon)

#alg = GeneralCausal()
#alg.run(200,model)

T_vals = range(model.K,1000,100)

algorithms = [GeneralCausal(),RandomArm(),ParallelCausal(),SuccessiveRejects()]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment = Experiment(5)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished,legend_loc = "lower left")
