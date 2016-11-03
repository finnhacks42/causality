# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:15:13 2016

@author: lat050
"""
import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def random_covariance(d,k):
    """ from http://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor """
    W = np.random.normal(size=(d,k))
    S = np.dot(W,W.T)+np.diag(np.random.uniform(size=d))
    return S


class Linear(object):
    def __init__(self,m,c = 0):
        " m is the number of features"
        self.w_t = np.random.normal(loc = .5,size = m+1)
        self.w_c = np.random.normal(loc = .5 + c*self.w_t,size = m+1)

        self.m = m
        self.ce = lambda x: expit(x.dot(self.w_t)) - expit(x.dot(self.w_c))

    def causal_effect(self,x):
        """ expected causal effect. input feature vector x"""
        x = np.hstack((x,np.ones((len(x),1))))
        return self.ce(x)

    def sample(self,n,prop_treat = .5):
        """"
        discrete: for each categorical feature a list of weights defining a multinomial distribution over the levels.
        """
        x = np.random.normal(size=(n,self.m+1)) #TODO make this be able to include dependencies - look into wishart distribution for covariance

        x[:,-1] = 1
        n_treat = int(n*prop_treat)
        x_treat = x[0:n_treat,:]
        x_control = x[n_treat:,:]
        self.p_treat = expit(x_treat.dot(self.w_t))
        self.y_treat = np.random.binomial(1,self.p_treat)
        self.p_control = expit(x_control.dot(self.w_c))
        self.y_control = np.random.binomial(1,self.p_control)
        return x_treat[:,:-1],x_control[:,:-1],self.y_treat,self.y_control

    def sample_x(self,n):
        return np.random.normal(size=(n,self.m))

class GCluster(object):
    """
    assumes the observable characteristics of each type of respondent (never, complier, defier, always)
    are sampled from a multivariate-gaussian distribution for that type.

    """

    def __init__(self, means, covariances, respondent_weights = [.3,.5,0,.2]):
        """
        means: a list of the means for each group (never takers, compliers, defiers, always takers).
        """

        self.respondent_weights = respondent_weights
        self.rvs = [ multivariate_normal(mean = m, cov = c) for (m,c) in zip(means,covariances)]
        self.group_names = ["never","comply","defy","always"]
        self.m = len(np.atleast_1d(means[0]))

    @classmethod
    def create2D(cls):
        """ create a simple instance in with features in R2 """
        means = [[1,1],[1,-1],[-1,-1],[-1,1]]
        cov = [np.identity(2) for i in range(4)]
        return cls(means,cov)

    @classmethod
    def create1D(cls):
        """ a simple instance with feature in R1 """
        means = [-1,1,-2,0]
        cov = [np.identity(1) for i in range(4)]
        return cls(means,cov)

    def plot1D(self,points = 100):
        # plot p(x) for each group
        f,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,10))
        x = np.linspace(-3,3,points)
        px = np.zeros(points)
        for i in range(4):
            px_i = self.respondent_weights[i]*self.rvs[i].pdf(x)
            px+=px_i
            ax[0].plot(x,px_i,label = self.group_names[i])
        ax[0].plot(x,px,label = "combined")
        ax[0].legend(loc = "upper left")
        py0gx,py1gx = self.p_of_y0_y1_given_x(x)
        ax[1].plot(x,py0gx,label="P(Y0)")
        ax[1].plot(x,py1gx,label="P(Y1)")
        ax[1].plot(x,py1gx-py0gx,label="CE")
        ax[1].legend(loc= "upper left")


    def p_of_group_given_x(self,x): # TODO check you haven't srewed this up (variables are continuous)
        # currently written for 1D x
        p_group_given_x = np.zeros((4,len(x)))
        for i in range(4):
            p_of_x_given_gi = self.rvs[i].pdf(x)
            p_group_given_x[i,:] = (p_of_x_given_gi*self.respondent_weights[i])
        p_group_given_x = 1.0/p_group_given_x.sum(axis=0)*p_group_given_x
        return p_group_given_x.T

    def p_of_y0_y1_given_x(self,x):
        pgx = self.p_of_group_given_x(x)
        py0,py1 = self.p_of_y0_y1_given_group(np.arange(4))
        py0gx = (py0*pgx).sum(axis=1)
        py1gx = (py1*pgx).sum(axis=1)
        return py0gx,py1gx

    def p_of_y0_y1_given_group(self,g):
        y0 = np.logical_or(g == 2, g==3).astype(int) # defiers and always have value 1 in control setting
        y1 = np.logical_or(g == 1, g==3).astype(int) # compliers and always have value 1 in treat setting
        return y0,y1



    def sample(self,n, prop_treat = 0.5):
        """ sample with treatment at random """
        group_counts = np.random.multinomial(n,self.respondent_weights)
        groups = np.repeat(range(4),group_counts)
        y0,y1 = self.p_of_y0_y1_given_group(groups)

        x = np.zeros((n,self.m))
        i = 0
        for group,count in enumerate(group_counts):
            x[i:i+count,:] = self.rvs[group].rvs(size = count).reshape((count,self.m))
            i = i+count

        ordering = range(n)
        np.random.shuffle(ordering)
        x = x[ordering]
        y0 = y0[ordering]
        y1 = y1[ordering]

        n_treat =int(n*prop_treat)

        x_treat = x[0:n_treat,:]
        y_treat = y1[0:n_treat]
        x_control = x[n_treat:,:]
        y_control = y0[n_treat:]
        return x_treat,x_control,y_treat,y_control








