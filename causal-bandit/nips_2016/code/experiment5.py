# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:44:51 2016

Compare the performance of the algorithms on a Confounded Parallel bandit for which 
there is no action to set Z. The Parallel 
"""


from models import ParallelConfoundedNoZAction
from algorithms import SuccessiveRejects,GeneralCausal,ParallelCausal
from experiment_config import now_string,Experiment
import numpy as np


def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    for T_indx,T in enumerate(T_vals): 
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
        print T
                
    return regret
           
                     
N = 20
pz = .5
q = (0,0,1,0)
epsilon = .1
simulations = 10
model = ParallelConfoundedNoZAction(*q,pZ=pz,N1=1,N2 = N-1,epsilon=epsilon)
print "model",model.m
T_vals = range(10,6*model.K,100)

algorithms = [SuccessiveRejects(),GeneralCausal(),ParallelCausal()]

regret = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment = Experiment(5)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished)
