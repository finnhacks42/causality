# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np
import matplotlib.pyplot as plt
from models import ParallelConfounded
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
    """ calculate mean and variance over simulations online """

    regret = np.zeros((len(algorithms,len(T_vals))))
    for indx,T in enumerate(T_vals):
        for a_indx,alg in enumerate(algorithms):
            regret[a_indx,indx] = alg.run(T,model)
    
    s = np.zeros((len(algorithms,len(T_vals))))
    m = regret
    mu = regret
    
    for k in xrange(1,simulations):
        for indx,T in enumerate(T_vals):
            for a_indx,alg in enumerate(algorithms):
                regret[a_indx,indx] = alg.run(T,model)
        m_next = m+(regret - m)/k
        s_next = s+(regret - m)*(regret - m_next)
        m = m_next
        s = s_next
        mu += regret
    
    mu = mu/simulations
    variance = s/simulations
    return mu,variance
    
    

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
    
def plot_regret(regret,xvals,xlabel,algorithms,detail_string,legend_loc = "upper right"):
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
    fig_name = "regret_vs_{0}_{1}_{2:.0f}".format(xlabel,detail_string,time.time())
    fig_name = fig_name.replace(".","-")+".png"
    fig.savefig(fig_name,bbox_inches="tight")
    
    
def regret_vs_m(model,T,algorithms,simulations=10):
    regret = np.zeros((len(algorithms),simulations))
    for s in xrange(simulations):
        for a_indx,alg in enumerate(algorithms):
            regret[a_indx,s] = alg.run(T,model)
    return regret
    
def experiment2_w(tpl):
    parameters,key_parameters = tpl
    return regret_vs_m(*parameters,**key_parameters)        
    
    
 
if __name__ == "__main__":  
    start = time.time()
    N = 20
    
    pz = .5
    q = (0.0001,0.0001,.8,.2)
    epsilon = .1
    simulations = 1000
    T= 400
    detail = "Tis{0}_qis{1}_pzis{2}_epsilonis_{3}_sims{4}_Nis{5}".format(T,q,pz,epsilon,simulations,N)
    
   
    algorithms = [AlphaUCB(2),SuccessiveRejects(),GeneralCausal()]
    m_vals = []
    regrets = []
    for N1 in range(1,20,2):
        model = ParallelConfounded.create(N,N1,pz,q,epsilon)
        m_vals.append(model.m)
        regret = regret_vs_m(model,T,algorithms,simulations = simulations)
        model.clean_up()
        regrets.append(regret)
    regrets = np.stack(regrets,axis=1)
    plot_regret(regrets,m_vals,"m",algorithms,detail)
    
        
        
    
    
    #regret = run_parallel_simulations(experiment2_w,simulations,mp.cpu_count(),[models,T,algorithms],{})    
    #plot_regret(regret,m_vals,"m",algorithms,detail)
    #regret = run_parallel_simulations(experiment2_w,simulations,mp.cpu_count(),[models,T,algorithms],{})
    #regret = run_parallel_simulations(experiment1_w,simulations,mp.cpu_count(),[model,T_vals,algorithms],{})
    #plot_regret(regret,T_vals,"T",algorithms,model)
    
    end = time.time()
    print end - start


    #TODO fix calculation of eta,m when probabilities are 0
    #TODO implement very confounded model
    #TODO improve optimisation
    #TODO calculate m for various model variants
    #TODO ensure parallel model compatible with new approach
    #TODO allow resetting of epsilon (and do variable epsilon version)











              
