# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:26:36 2016

@author: finn
"""

from cvxpy import *

x = Variable(2)
obj = Minimize(x[0] + norm(x, 1))
constraints = [x >= 2]
prob = Problem(obj, constraints)
prob.solve()