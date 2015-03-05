from math import sqrt,log
from scipy.stats import bernoulli
import numpy as np
class BernoulliEstimator(object):
    """ records number of success,failures and calculates bounds and expectation"""
    def __init__(self,success,fail,alpha):
        self.success = success
        self.fail = fail
        self.alpha = alpha
        self.n = float(self.success+self.fail)
        
    def upper(self,t):
        """ returns Bubeck's UCB upper bound"""
        return self.mean()+sqrt(self.alpha*log(t+1)/(2.0*self.n))
        
    def mean(self):
        return self.success/self.n
        
    def update(self,reward):
        """ reward should be 1 for success and 0 for fail"""
        self.success+=reward
        self.fail+= (1-reward)
        self.n +=1

   
def ucbBound(model,alpha,horizon):
    t = np.arange(horizon)
    rewards = [model.expectedYGivenAction(a) for a in range(model.numArms)]
    inverseDeltas = []
    for r in rewards:
        deltaI = model.best - r
        if deltaI > 0:
            inverseDeltas.append(1.0/deltaI)
    s = sum(inverseDeltas)
    bound = 2*alpha*np.log(t)*s+len(inverseDeltas)*alpha/(alpha-2)
    return bound
    
            

class TrivialProbabilityModel(object):
    """ Holds only for the special case where each X variable is an independent cause of Y. 
        1) p(x1)...p(xn) = 0.5
        2) Y is independent of all x except x1.
        3) P(y|x1=1) = 0.5+epsilon, P(y|x1=0) = 0.5-epsilon
        4) => P(y|xi=j) = 0.5 for all other variables"""
    def __init__(self,numVars,epsilon):
        self.numVars = numVars
        self.epsilon = epsilon
        self.px = bernoulli(.5)
        self.pyx1 = [bernoulli(0.5-epsilon),bernoulli(0.5+epsilon)]
        self.py = bernoulli(0.5)
        self.best = 0.5+epsilon
        self.numArms = numVars*2
        

    def sampleX(self):
        """ sample a vector X from P(X)"""
        return self.px.rvs(size=self.numVars)
        
    def sampleYGivenX(self,X):
        """ sample from P(Y|X), where X is the full vector of variables values"""
        x1 = X[0]
        return self.pyx1[x1].rvs()

    def expectedYGivenAction(self,actionID):
        """ returns the expected reward of the action specified by the given index """
        if actionID == 0:
            return 0.5-self.epsilon
        elif actionID == 1:
            return 0.5+self.epsilon
        else:
            return 0.5

    def expectedY(self):
        """returns P(Y) (after marginalizing out all the X's)"""
        return 0.5

    def toVar(self,actionID):
        """action indecies correspond to x1=0,x1=1,x2=0,x2=1 ... xn=0,xn=1 """
        var = actionID/2
        value = actionID % 2
        return (var,value)

    def toID(self,var,value):
        return int(var)*2+int(value)

    def sampleYGivenXi(self,var,value):
        """ sample from P(Y|X_i), where X_i is the value of the ith-variable (ie marginalize out the full conditional)"""
        if var == 0:
            return self.pyx1[value].rvs()
        else:
            return self.py.rvs()

    def sampleY(self,actionID):
        action = self.toVar(actionID)
        return self.sampleYGivenXi(*action)
    
    
