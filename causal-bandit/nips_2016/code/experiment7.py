# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:19:07 2016

@author: finn
"""

from models import ScaleableParallelConfounded
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
           



                  
N = 50
N1 = 1
pz = .3
q = (.1,.9,.2,.7) #q = (.3,.3,.4,.6)
epsilon = .2
pY = ScaleableParallelConfounded.pY_epsilon_best(q,pz,epsilon)

simulations = 1000
#
model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)
model.compute_m()
model.make_ith_arm_epsilon_best(epsilon,N)


#
T_vals = range(25,451,25)

algorithms = [GeneralCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment = Experiment(7)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished,legend_loc = None)