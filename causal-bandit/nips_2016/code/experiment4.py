# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from models import ParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal
from experiment_config import now_string,Experiment
import numpy as np


def regret_vs_m_general(algorithms,N1_vals,N,T,pz,q,epsilon,simulations = 10): 
    m_vals = []
    regret = np.zeros((len(algorithms),len(N1_vals),simulations))
    for m_indx,N1 in enumerate(N1_vals):
        model = ParallelConfounded.create(N,N1,pz,q,epsilon)
        print N1
        m_vals.append(model.m)
        for a_indx, algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
    
    return m_vals,regret
    
   

    
N = 20
N1_vals = range(0,N,2)
pz = .5
q = (0,0,1,0)
epsilon = .1
simulations = 10000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal()]


m_vals,regret = regret_vs_m_general(algorithms,N1_vals,N,T,pz,q,epsilon,simulations=simulations)
finished = now_string()

experiment = Experiment(4)
experiment.plot_regret(regret,m_vals,"m",algorithms,finished,legend_loc = "lower right")
experiment.log_regret(regret,finished)
experiment.log_code(finished)

