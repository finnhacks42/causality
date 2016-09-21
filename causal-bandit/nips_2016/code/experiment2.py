# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:30:34 2016

@author: finn
"""
import numpy as np
from math import sqrt,ceil
from models import Parallel
from algorithms import GeneralCausal, ParallelCausal, SuccessiveRejects
from experiment_config import now_string, Experiment


def regret_vs_T_vary_epsilon(model,algorithms,simulations = 10):

    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    
    for T_indx,T in enumerate(T_vals): 
        print T
        epsilon = sqrt(model.K/(a*T))
        model.set_epsilon(epsilon)
        for s in xrange(simulations):
            for a_indx, algorithm in enumerate(algorithms):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
        
    return regret

N= 50
simulations = 10
a = 9.0
m = 2
model = Parallel.create(N,m,.1)

Tmin = int(ceil(4*model.K/a))
Tmax = 10*model.K
T_vals = range(Tmin,Tmax,10)

algorithms = [GeneralCausal(),ParallelCausal(),SuccessiveRejects()]

regret = regret_vs_T_vary_epsilon(model,algorithms,simulations = simulations)
finished = now_string()

experiment = Experiment(2)
experiment.plot_regret(regret,T_vals,"T",algorithms,finished)
experiment.log_code(finished)
experiment.log_regret(regret,finished)




#regret2,mean2,error2 = regret_vs_T(N,simulations,epsilon=None,Tstep=None,TperK=10)
#pickle.dump(regret2, open("experiment2_{0}.pickle".format(now_string()),"wb"))