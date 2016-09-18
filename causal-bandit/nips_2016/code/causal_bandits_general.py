# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models import GeneralModel, Parallel, ParallelConfounded
from algorithms import GeneralCausal,  ParallelCausal, SuccessiveRejects, LilUCB

REGRET_LABEL = "Regret"
HORIZON_LABEL = "T"
M_LABEL = "m(q)"

T_vals = range(50,551,100)
#confounded_parallel = GeneralModel.create_confounded_parallel(3,.1)

#model1 = Parallel.create(10,.1)
model2 = ParallelConfounded.create(5,.1)


alg1 = ParallelCausal()
alg2 = GeneralCausal()
sr = SuccessiveRejects()


eta,m = alg2.find_eta(model2)


regret = np.zeros((2,len(T_vals)))
for indx,T in enumerate(T_vals):
    print(T)
    regret[0,indx] = sr.run(T,model2)
    regret[1,indx] = alg2.run(T,model2,eta,m)
        





#result_alg1 = alg1.run(T,model)

#result_sr = sr.run(T,model)





              
