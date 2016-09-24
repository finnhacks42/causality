# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:44:51 2016

Compare the performance of the algorithms on a Confounded Parallel bandit for which 
there is no action to set Z. The Parallel algorithm is mis-specified in this setting, as it assumes there are no confounders.
If the resulting bias exceeds epsilon then the Parallel algorithm will never identify the best arm.
"""


from models import ParallelConfoundedNoZAction,ParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,ParallelCausal
from experiment_config import now_string,Experiment
import numpy as np
from pgmpy_model import GeneralModel


def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    for T_indx,T in enumerate(T_vals): 
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
        print T
                
    return regret
           
simulations = 100  
epsilon = .2

                  
#N = 20
#N1=1
#pz = .3
#q = (.9,.1,.8,.1)
#model = ParallelConfoundedNoZAction.create(N,N1,pz,q,epsilon)

N = 5
N1 = 1
pz = .5
q = (.9,.1,.9,.1)
epsilon = .1

model = GeneralModel.create_confounded_parallel(N,N1,pz,q,epsilon,act_on_z=False)





T_vals = range(10,1000,200)

algorithms = [SuccessiveRejects(),GeneralCausal(),ParallelCausal()]

regret = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment = Experiment(5)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished)
