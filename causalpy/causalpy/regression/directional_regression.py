# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:26:14 2016
The idea is that we assume human experts know which featues will have 'substantial' impact and the direction of the
impact for those features.
@author: finn
"""

#from pgmpy.models import LinearGaussianBayesianNetwork
#from pgmpy.factors.continuous import LinearGaussianCPD, JointGaussianDistribution, ContinuousFactor
#from scipy.special import beta as beta_func
#from scipy.stats import beta
#from scipy.stats import multivariate_normal
import numpy as np

from sympy import Symbol, Matrix

class LinearGaussianBN(object):
    def __init__(self):
        self.variables = []
        self.means = Matrix([])
        self.cov = Matrix([])

    def add_cpd(self,variable,parents = [], variance = None):
        if variance is None:
            variance = Symbol("V[{0}]".format(variable))
        beta = [Symbol("w_"+variable+v) if v in parents else 0 for v in self.variables]
        mu = Symbol("w_"+variable+"0")
        v = variance

        if len(beta) > 0:
            beta = Matrix(beta)

            mu +=(beta.T*self.means)[0,0]
            cv = self.cov*beta # covariance of this variable with previous variables
            v += (beta.T*cv)[0,0] # variance of this variable (unconditional)

            new_vals = Matrix([cv,[v]])
            rows,cols = self.cov.shape
            self.cov = self.cov.row_insert(rows,cv.T)
            self.cov = self.cov.col_insert(cols,new_vals)


        else: # first time round - everything is empty
            self.cov = Matrix([v])


        self.means = Matrix([self.means,[mu]])
        print(self.cov)

        self.variables.append(variable)

    def marginalize(self,variables):
        """ returns a new network with the specified variables marginalized out """
        return

    def observe(self,variable,value):
        """ network where we condition on variable equaling some value """
        return

    def do(self,variable,value):
        """ network after intervening to set specified variable to value """
        return

model = LinearGaussianBN()
model.add_cpd('z')
model.add_cpd('x',['z'])
model.add_cpd('y',['z','x'])

#n = 20
#z = np.random.multivariate_normal(mean = [0,0],cov = np.identity(2),size = n)
#w_x = np.asarray([[0,-2],[.5,-.3],[.6,.1]]) # weights for x, row 1 is offsets, row 2 is with respect to z1, row 3 with respect to z2
#x = np.dot(np.hstack((np.ones((n,1)),z)),w_x) + np.random.normal(scale = .5,size = n)
#w_y = np.asarray([0,1,1,]) # offset,w_z1,w_z2,w_x1,w_x2






#model = LinearGaussianBayesianNetwork([('z','x'),('x','y'),('z','y')])
#cpd_z = LinearGaussianCPD('z',1,beta_vector=[4]) # P(z) = N(1,4)
#cpd_x = LinearGaussianCPD('x',3,['z'],[-2,.5]) #P(x|z) = N(.5z - 2,3)
#cpd_y = LinearGaussianCPD('y',1,['z','x'],[0,.8,2]) #P(y|x,z) = N(.8z+2x,1)
#model.add_cpds(cpd_z, cpd_x, cpd_y)
#model.check_model()
## I wonder if this ever worked ...
#g = model.to_joint_gaussian()

#normal_pdf = lambda x1, x2: multivariate_normal.pdf((x1, x2), [0, 0], [[1, 0], [0, 1]])
#phi = ContinuousFactor(['x1', 'x2'], normal_pdf)
#phi2 = ContinuousFactor(['x1', 'x2'],multivariate_normal([0, 0], [[1, 0], [0, 1]]))
#phi3 = ContinuousFactor(['x1', 'x2'],'multivariate_normal', mean=[0, 0], cov=[[1, 0], [0, 1]])
#print(phi.assignment(1, 2))
#print(phi2.assignment(1, 2))
#print(phi3.assignment(1,2))
#
#
#df2 = ContinuousFactor(['x','y'],'dirichlet',alpha=[1,2])
#
##>>> dirichlet_factor = ContinuousFactor(['x', 'y'], drichlet_pdf)
##        >>> dirichlet_factor.scope()
##        ['x', 'y']
##        >>> dirichlet_factor.assignemnt(5,6)
##        226800.0
#
#
#c1 = ContinuousFactor(['x'],beta(3,1))
#c2 = ContinuousFactor(['x'],'beta',a=3,b=1)
#print(c1.assignment(.2))
#print(c2.assignment(.2))

# think about if the way arguaments are getting passed is actually optimal (to pdf)







