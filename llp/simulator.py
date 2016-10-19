# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:15:13 2016

@author: lat050
"""
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import minimize



class Linear(object):
    def __init__(self,m,c = 0):
        self.w_t = np.random.normal(loc = .5,size = m+1)
        self.w_c = np.random.normal(loc = .5 + c*self.w_t,size = m+1)
        
        self.m = m
        self.ce = lambda x: expit(x.dot(self.w_t)) - expit(x.dot(self.w_c))
    
    def causal_effect(self,x):
        """ expected causal effect. input feature vector x, last element 1 to handle intercept """
        x = np.hstack((x,np.ones((len(x),1))))
        return self.ce(x)
        
    def sample(self,n,prop_treat = .5):
        x = np.random.normal(size=(n,self.m+1))
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

  
class StandardEstimator(object):
    label = "standard"
    def __init__(self):
        self.model_t = LogisticRegression()
        self.model_c = LogisticRegression()
    
    def fit(self,x_treat,x_control,y_treat,y_control):
        self.model_t.fit(x_treat,y_treat)
        self.model_c.fit(x_control,y_control)
        self.w_t = np.hstack((self.model_t.coef_[0],self.model_t.intercept_))
        self.w_c = np.hstack((self.model_c.coef_[0],self.model_c.intercept_))
    
    def predict(self,x):
        return self.model_t.predict_proba(x)[:,1] - self.model_c.predict_proba(x)[:,1]

class SoftLogisticRegression(object):
    """ 
    Logistic regression that takes class probabilities rather than labels as inputs.
    Minimizes the cross entropy.
    """
    
    def fit(self,x,p):
        """ p is a vector of the probability y_i = 1 """
        assert x.ndim == 2, "x should be a 2d numpy array"
        assert p.ndim == 1, "p should be a 1d numpy array, not ndims:{0}, shape:{1}".format(p.ndim,p.shape)
        assert x.shape[0] == p.shape[0], "number of elements in p: {0}, must match number of rows of x: {1}".format(p.shape[0],x.shape[0])
        x2 = np.hstack((x,np.ones((x.shape[0],1))))
        w0 = np.random.random((x2.shape[1],1))
        optimal = minimize(self._loss,w0,args = (x2,p),jac = self._jac,options = {"disp":False})
        self.w = optimal.x
        self.coef_ = optimal.x[0:-1]
        self.intercept_ = optimal.x[-1]
        
    def predict_proba(self,x):
        x2 = np.hstack((x,np.ones((x.shape[0],1))))
        return expit(x2.dot(self.w))
    
    def _loss(self,w,x,p):
        q = expit(x.dot(w))
        l_i = -(p*np.log(q) + (1.0-p)*np.log(1-q))
        return l_i.sum()
    
    def _jac(self,w,x,p):
        """ the gradient of the loss function """
        xw = x.dot(w)
        q = expit(xw)
        return -(p*q*np.exp(-xw)-(1-p)*(1-q)*np.exp(xw)).dot(x)
    
    @staticmethod           
    def soft_equiv(y):
        """ p is 1 if y is 1, 0 otherwise """
        p = y.copy()
        p[y != 1] = 0
        return p
        
                
class EMEstimator(object):
    label = "EM"
    
    def fit(self,x_treat,x_control,y_treat,y_control, tol = 1e-2, max_iter = 500):
        base_estimator = StandardEstimator()
        base_estimator.fit(x_treat,x_control,y_treat,y_control)
        model_t = SoftLogisticRegression()
        model_c = SoftLogisticRegression()
        
        p_ct = base_estimator.model_t.predict_proba(x_control)[:,1] # probability p(y=1|x,a=1) for x in X_c
        p_tc = base_estimator.model_c.predict_proba(x_treat)[:,1]   # probability p(y=1|x,a=0) for x in X_t
        
        
        p_cc = SoftLogisticRegression.soft_equiv(y_control) # note these are fixed and not updated
        p_tt = SoftLogisticRegression.soft_equiv(y_treat)
        
        x = np.vstack((x_control,x_treat))
        w_c = base_estimator.w_c
        w_t = base_estimator.w_t
        
        
        i = 0
        while (i < max_iter):
            
            p_c = np.concatenate((p_cc,p_tc)) #results for control, treated if a = 0
            p_t = np.concatenate((p_ct,p_tt)) # results for control,treated if a = 1
            
            model_c.fit(x,p_c)
            model_t.fit(x,p_t)
            
            delta_c = mean_squared_error(model_c.w, w_c)
            delta_t = mean_squared_error(model_t.w , w_t)
            
            if  delta_c < tol and delta_t < tol:
                break
            
            #print w_c,delta_c,delta_t
                                
            w_c = model_c.w
            w_t = model_t.w
            
            p_ct = model_t.predict_proba(x_control)
            p_tc = model_c.predict_proba(x_treat)
            
            i += 1
        
        self.model_t = model_t
        self.model_c = model_c
        if i == max_iter:
            message = "failed to coverge to requested tolerance within max_iter={0} iterations.\n deltas: {5}, {6}\n{1}\n{2}\n\n{3}\n{4}".format(max_iter,w_c,model_c.w,w_t,model_t.w,delta_c,delta_t)
            raise Exception(message)
        
        
        
    def predict(self,x):
        return self.model_t.predict_proba(x) - self.model_c.predict_proba(x)
        
def plot_single_feature(sample_size):
    simulator = Linear(1,c=0)
    x_treat,x_control,y_treat,y_control = simulator.sample(sample_size)

    estimator = StandardEstimator()
    estimator.fit(x_treat,x_control,y_treat,y_control)
    
    estimator2 = EMEstimator()
    estimator2.fit(x_treat,x_control,y_treat,y_control,tol=1e-3)
     
    x_vals = np.linspace(-4,4,num = 50).reshape(50,1)
    
    plt.plot(x_vals,simulator.causal_effect(x_vals),label = "true")
    plt.plot(x_vals,estimator.predict(x_vals),label = estimator.label)
    plt.plot(x_vals,estimator2.predict(x_vals),label = estimator2.label)
    plt.legend(loc = "lower right")
    

def compare_errors(n_features, nrows ,simulations, c_vals = [0,.5,1],test_samples = 1000):
    scores = np.zeros((2,len(c_vals),simulations))    
    for c_indx,c in enumerate(c_vals):
        for s in xrange(simulations):
            model = Linear(n_features, c = c)
            print c,s,model.w_c,model.w_t
            x_treat,x_control,y_treat,y_control = model.sample(nrows)
            estimator = StandardEstimator()
            estimator.fit(x_treat,x_control,y_treat,y_control)
            estimator2 = EMEstimator()
            estimator2.fit(x_treat,x_control,y_treat,y_control)
            x_test = model.sample_x(test_samples)
            ce1 = estimator.predict(x_test)
            ce2 = estimator2.predict(x_test)
            ce_true = model.causal_effect(x_test)
            scores[0,c_indx,s] = mean_squared_error(ce_true,ce1)
            scores[1,c_indx,s] = mean_squared_error(ce_true,ce2)
    return scores
        
        
    
        
np.random.seed(1)  

scores = compare_errors(1,200,100)

#plot_single_feature(200)
   



        




#sim = Linear(3)
#x_treat,x_control,y_treat,y_control = sim.sample(100)
#p_treat = sim.p_treat
#
#m1 = LogisticRegression()
#m1.fit(x_treat,y_treat)
#
#m2 = SoftLogisticRegression()
#m2.fit(x_treat,p_treat)
#
#m3 = SoftLogisticRegression()
#m3.fit(x_treat,m3.soft_equiv(y_treat))
#
#print sim.w1
#print m2.coef_,m2.intercept_
#
#print m1.coef_,m1.intercept_
#print m3.coef_,m3.intercept_
#        
    