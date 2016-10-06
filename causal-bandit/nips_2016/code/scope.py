# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 07:41:12 2016

@author: finn
"""

from experiment_config import Experiment

e = Experiment(1)
def returnthing():
    return 5

v1 = 4
v2 = [1,2,3]
x = returnthing()

e.log_state(globals())

del v1
del v2
del x

d = e.read_state(e.state_filename)

#import shelve
#d = {}
#db = shelve.open(e.state_filename)
#for key in db:
#    
#    value = db[key]
#    d[key] = value
#    globals()[key] = value
#    print key,value
#db.close()

