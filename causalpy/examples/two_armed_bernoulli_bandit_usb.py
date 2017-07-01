# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:39:57 2017

@author: finn
"""

from causalpy.bandits import BasicBernoulliModel, AlphaUCB
import numpy as np
import matplotlib.pyplot as plt


simulations = 1000
T = 10

delta = .3
expected_rewards = [.5-delta/2.0,.5+delta/2.0]
enviroment = BasicBernoulliModel(expected_rewards)
algorithm = AlphaUCB(4.0)

algorithm.run(T,enviroment)
regret = algorithm.get_cumulative_regret()

#for i,T in enumerate(T_vals):
#    enviroment = BasicBernoulliModel.create_epsilon_best(K,.1)
#    best_reward = np.empty(simulations)
#    expected_best_is_best = 0
#    for s in xrange(simulations):
#        rewards = enviroment.sample_multiple(np.arange(0,K),T)
#        best_arm = rewards.argmax()
#        best = rewards[best_arm]
#        best_reward[s] = best
#        expected_best_is_best += int(best_arm == (K-1))
#        
#    
#    max_expected_reward = enviroment.expected_rewards.max()*T  
#    average_max_reward = best_reward.mean()
#    result[i,0] = max_expected_reward
#    result[i,1] = average_max_reward
#    result[i,2] = expected_best_is_best/float(s)
    