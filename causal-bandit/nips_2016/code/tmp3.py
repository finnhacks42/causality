# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:04:42 2016

@author: finn
"""
from experiment_config import Experiment
e = Experiment(1)
globals().update(e.read_state("results/experiment_state1_20161007_0842.shelve"))


    

#db = shelve.open("a.shelve")
#db["a"] = 4
#db["b"] = [1,2,3]
#for key in db:
#    print key
#db.close()
#
#print "newly opened"
#db = shelve.open("a.shelve")
#for key in db:
#    print key
#db.close()
#

