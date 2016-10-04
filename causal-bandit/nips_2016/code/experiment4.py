# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from models import ParallelConfounded,ScaleableParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,AlphaUCB,ThompsonSampling
from experiment_config import now_string,Experiment
import numpy as np


def regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = 1000): 
    m_vals = []
    regret = np.zeros((len(algorithms),len(N1_vals),simulations))
    for m_indx,N1 in enumerate(N1_vals):
        model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)
        #eta = [0,0,1.0/(N1+6.0),0,0,0,1-N1/(N1+6.0)]
        model.compute_m()
        
        print N1,model.m
        m_vals.append(model.m)
        for a_indx, algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                i = np.random.randint(0,model.K)
                model.make_ith_arm_epsilon_best(epsilon,i)
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
    
    return m_vals,regret
    
   

    
N = 50
N1_vals = range(1,N,2)
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .1
simulations = 1000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal(),AlphaUCB(2),ThompsonSampling()]
pY = np.asarray([[.4,.4],[.7,.7]])


m_vals,regret = regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = simulations)
finished = now_string()

experiment = Experiment(4)
experiment.plot_regret(regret,m_vals,"m",algorithms,finished,legend_loc = "lower right")
experiment.log_regret(regret,finished)
experiment.log_code(finished)

