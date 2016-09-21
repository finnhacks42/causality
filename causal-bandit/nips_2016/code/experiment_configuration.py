# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:18:04 2016

@author: finn
"""

import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from algorithms import GeneralCausal,  ParallelCausal, SuccessiveRejects,AlphaUCB


def now_string():
    return dt.datetime.now().strftime('%Y%m%d_%H%M')

def log_code(experiment,now_string):
    out = "experiment{0}_{1}_settings.txt".format(experiment,now_string)
    experiment_file = "experiment{0}.py".format(experiment)
    with open(experiment_file,"r") as f, open(out,"w") as o:
        o.write(f.read())
        
class Experiment(object):
    def __init__(self):
        self.REGRET_LABEL = "Regret"
        self.HORIZON_LABEL = "T"
        self.M_LABEL = "m(q)"
        markers=['s', 'o', 'D', '*']
        colors = ["red","green","blue","purple"]
        algorithms = [ParallelCausal,GeneralCausal,SuccessiveRejects,AlphaUCB]
        for indx,a in enumerate(algorithms):
            a.marker = markers[indx]
            a.color = colors[indx]
              
    def plot_regret(self,regret,xvals,xlabel,algorithms,experiment_number,now_string,legend_loc = "upper right"):
        s_axis = regret.ndim - 1 # assumes the last axis is over simulations
        simulations = regret.shape[-1]
        mu = regret.mean(s_axis)
        error = 3*regret.std(s_axis)/np.sqrt(simulations)
        fig,ax = plt.subplots()
        for indx,alg in enumerate(algorithms):    
            ax.errorbar(xvals,mu[indx,:],yerr = error[indx,:],label = alg.label,linestyle="",marker = alg.marker,color=alg.color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.REGRET_LABEL)
        
        ax.legend(loc = legend_loc,numpoints=1)
        fig_name = "experiment{0}_{1}.png".format(experiment_number,now_string)
        fig.savefig(fig_name,bbox_inches="tight")
        

            


 

    

    

    