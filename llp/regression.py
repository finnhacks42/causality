# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:21:50 2016
Explores how you interpret regression coefficients

Regularization makes sense for prediction, but does it make sense for parameter estimation?
Can you cross-validate?


@author: finn
"""

from sklearn.linear_model import LinearRegression
import numpy as np

#n = 10000
#
#u = 1+np.random.normal(scale = 1,size=n)
#x1 = .5 + 2*u+np.random.normal(scale = .2,size=n)
#x2 = -1 + 1*u+np.random.normal(scale = .1,size=n)
#x3 = 3+np.random.normal(scale=.4,size=n)
#t = -1+2*x3+np.random.normal(scale = .2,size=n)
#
#x = np.vstack((x1,x2,x3,t)).T
#w = np.asarray([.1,.4,-.6,.3])
#
#y = x.dot(w)+np.random.normal(scale=.2,size = n)
#model = LinearRegression()
#model.fit(x,y)
#print model.coef_


x = np.linspace(-3,3,500)

from scipy.stats import norm
import matplotlib.pyplot as plt

n1 = norm(loc=1,scale = .2)
n2 = norm(loc=0,scale=.5)
plt.plot(x,n1.pdf(x))
plt.plot(x,n2.pdf(x))
plt.plot(x,np.true_divide(n1.pdf(x),n2.pdf(x)))

