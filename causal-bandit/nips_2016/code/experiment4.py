# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from models import ParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,AlphaUCB
from experiment_config import now_string,Experiment
import numpy as np


def regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = 1000): 
    m_vals = []
    regret = np.zeros((len(algorithms),len(N1_vals),simulations))
    for m_indx,N1 in enumerate(N1_vals):
        model = ParallelConfounded.create(N,N1,pz,pY,q,epsilon)
        model.make_ith_arm_epsilon_best(epsilon,0)
        print N1
        m_vals.append(model.m)
        for a_indx, algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
    
    return m_vals,regret
    
   

    
N = 20
N1_vals = range(1,N,2)
pz = .4
q = (0,0,.4,.6)
epsilon = .1
simulations = 10000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal(),AlphaUCB(2)]
pY = np.asarray([[.4,.4],[.7,.7]])


m_vals,regret = regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = simulations)
finished = now_string()

experiment = Experiment(4)
experiment.plot_regret(regret,m_vals,"m",algorithms,finished,legend_loc = "lower right")
experiment.log_regret(regret,finished)
experiment.log_code(finished)

