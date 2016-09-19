# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np
import matplotlib.pyplot as plt
from models import GeneralModel, Parallel, ParallelConfounded
from algorithms import GeneralCausal,  ParallelCausal, SuccessiveRejects, LilUCB,AlphaUCB
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

REGRET_LABEL = "Regret"
HORIZON_LABEL = "T"
M_LABEL = "m(q)"

markers=['s', 'o', 'D', '*']
colors = ["red","green","blue","purple"]
algorithms = [ParallelCausal,GeneralCausal,SuccessiveRejects,AlphaUCB]
for indx,a in enumerate(algorithms):
    a.marker = markers[indx]
    a.color = colors[indx]
    

def experiment1(model,T_vals,algorithms,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    for indx,T in enumerate(T_vals):
        for s in xrange(simulations):
            for a_indx,alg in enumerate(algorithms):
                regret[a_indx,indx,s] = alg.run(T,model)
    return regret
        
    
def experiment1_w(tpl):
    parameters,key_parameters = tpl
    return experiment1(*parameters,**key_parameters)


def run_parallel_simulations(experiment, simulations, processes, parameters, key_parameters):
    """ assumes the experiment function takes a keyword arguament named 'simulations' and returns a numpy array with the simulations being the final dimension"""
    key_parameters["simulations"] = simulations/processes
    p = mp.Pool(processes = processes)
    tasks = [(parameters,key_parameters) for i in xrange(processes)]
    results = p.map_async(experiment,tasks).get()
    merged = np.concatenate(results,axis=results[0].ndim - 1)
    return merged
    
def plot_regret(regret,xvals,xlabel,algorithms,model,legend_loc = "upper right"):
    s_axis = regret.ndim - 1 # assumes the last axis is over simulations
    simulations = regret.shape[-1]
    mu = regret.mean(s_axis)
    error = 3*regret.std(s_axis)/np.sqrt(simulations)
    fig,ax = plt.subplots()
    for indx,alg in enumerate(algorithms):    
        ax.errorbar(xvals,mu[indx,:],yerr = error[indx,:],label = alg.label,linestyle="",marker = alg.marker,color=alg.color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(REGRET_LABEL)
    
    ax.legend(loc = legend_loc,numpoints=1)
    fig_name = "regret_vs_{0}_{1!s}_{2:.0f}".format(xlabel,model,time.time())
    fig.savefig(fig_name,bbox_inches="tight")
    
 
if __name__ == "__main__":  
    start = time.time()
    N = 20
    N1 = 2
    pz = .1
    q = (.1,.3,.4,.7)
    epsilon = .1
    simulations = 1000
    model = ParallelConfounded.create(N,N1,pz,q,epsilon)

    T_vals = range(10,500,50)
    algorithms = [AlphaUCB(2),SuccessiveRejects(),GeneralCausal()]
   
    regret = run_parallel_simulations(experiment1_w,simulations,mp.cpu_count(),[model,T_vals,algorithms],{})
    plot_regret(regret,T_vals,"T",algorithms,model)
    
    end = time.time()
    print end - start













              
