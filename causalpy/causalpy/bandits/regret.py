# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:57:12 2017

@author: finn
"""

def regret(algorithm,enviroment):
    """ standard cumulative regret for a particular run of an algorithm. """
    selected_arms = algorithm.trials # the counts of trials for each arm. 
    T = selected_arms.sum() 
    expected_algorithm_reward = enviroment.expected_rewards 
    best_reward = T*enviroment.expected_rewards.max()
    
    