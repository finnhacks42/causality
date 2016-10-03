# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:18:04 2016

@author: finn
"""

import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from algorithms import GeneralCausal,  ParallelCausal, SuccessiveRejects,AlphaUCB,RandomArm,ThompsonSampling
import cPickle as pickle

print "LOADING EXPERIMENT CONFIG MODULE AGAIN"

def now_string():
    return dt.datetime.now().strftime('%Y%m%d_%H%M')

  
class Experiment(object):
    
    def __init__(self,experiment_id):
        self.experiment_id = experiment_id
        self.REGRET_LABEL = "Regret"
        self.HORIZON_LABEL = "T"
        self.M_LABEL = "m(q)"
        self.markers=['s', 'o', 'D', '*',"^","p"]
        self.colors = ["red","green","blue","purple","cyan","orange"]
        self.algorithms = [ParallelCausal,GeneralCausal,SuccessiveRejects,AlphaUCB,RandomArm,ThompsonSampling]

        for indx,a in enumerate(self.algorithms):
            a.marker = self.markers[indx]
            a.color = self.colors[indx]
              
    def plot_regret(self,regret,xvals,xlabel,algorithms,now_string,legend_loc = "upper right"):
        s_axis = regret.ndim - 1 # assumes the last axis is over simulations
        simulations = regret.shape[-1]
        mu = regret.mean(s_axis)
        error = 3*regret.std(s_axis)/np.sqrt(simulations)
        fig,ax = plt.subplots()
        for indx,alg in enumerate(algorithms):    
            ax.errorbar(xvals,mu[indx,:],yerr = error[indx,:],label = alg.label,linestyle="",marker = alg.marker,color=alg.color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.REGRET_LABEL)
        
        if legend_loc is not None:
            ax.legend(loc = legend_loc,numpoints=1)
        fig_name = "results/experiment{0}_{1}.pdf".format(self.experiment_id,now_string)
        fig.savefig(fig_name,bbox_inches="tight")
    
    def log_code(self,now_string):
        out = "results/experiment{0}_{1}_settings.txt".format(self.experiment_id,now_string)
        experiment_file = "experiment{0}.py".format(self.experiment_id)
        with open(experiment_file,"r") as f, open(out,"w") as o:
            o.write(f.read())
    
    def log_regret(self,regret,now_string):
        filename = "results/experiment{0}_{1}.pickle".format(self.experiment_id,now_string)
        with open(filename,'wb') as out:
            pickle.dump(regret,out)
        
        
    
        

            


 

    

    

    