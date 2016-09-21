# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:47:47 2016

@author: finn
"""
import numpy as np
from models import Parallel
from algorithms import GeneralCausal, ParallelCausal, SuccessiveRejects
from experiment_config import now_string, Experiment


def regret_vs_m(algorithms,m_vals,N,T,epsilon,simulations = 10):  
    regret = np.zeros((len(algorithms),len(m_vals),simulations))
    for m_indx,m in enumerate(m_vals):
        model = Parallel.create(N,m,epsilon)
        print "built model {0}".format(m)
        for s in xrange(simulations):
            for a_indx,algorithm in enumerate(algorithms):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
    
    return regret

experiment = Experiment(1)
          
# Experiment 1
N = 50
epsilon = .3
simulations = 100
T = 400
algorithms = [GeneralCausal(),ParallelCausal(),SuccessiveRejects()]
m_vals = range(2,N,2)
    
regret = regret_vs_m(algorithms,m_vals,N,T,epsilon,simulations = simulations)
finished = now_string()

experiment.plot_regret(regret,m_vals,"m",algorithms,finished)
experiment.log_code(finished)
experiment.log_regret(regret,finished)
