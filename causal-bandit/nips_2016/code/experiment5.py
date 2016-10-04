# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:44:51 2016

Compare the performance of the algorithms on a Confounded Parallel bandit for which 
there is no action to set Z. The Parallel algorithm is mis-specified in this setting, as it assumes there are no confounders.
If the resulting bias exceeds epsilon then the Parallel algorithm will never identify the best arm.
"""


from models import ScaleableParallelConfoundedNoZAction, ParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,ParallelCausal,RandomArm,AlphaUCB,ThompsonSampling
from experiment_config import now_string,Experiment
import numpy as np



def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    pulls = np.zeros((len(algorithms),len(T_vals),model.K),dtype=int)
    for T_indx,T in enumerate(T_vals): 
        
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
                if algorithm.best_action is not None:
                    pulls[a_indx,T_indx,algorithm.best_action] +=1
        print T
                
    return regret,pulls
           
simulations = 10000

                  
N =50
N1 = 1
pz = .4
q = (.1,.9,.2,.7)
epsilon = .2
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

model = ScaleableParallelConfoundedNoZAction(q,pz,pY,N1,N-N1)
model.compute_m()
model.make_ith_arm_epsilon_best(epsilon,N)

#alg = GeneralCausal()
#alg.run(200,model)

T_vals = range(25,451,50)
#
algorithms = [GeneralCausal(),ParallelCausal(),SuccessiveRejects(),ThompsonSampling(),AlphaUCB(2)]
#
regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment = Experiment(5)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished,legend_loc = "lower left")
