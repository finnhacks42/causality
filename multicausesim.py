import numpy as np
from scipy.stats import bernoulli
from scipy.stats import beta
from itertools import izip
from random import randint
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def generateCombinations(nVars):
    rows = pow(2,nVars)
    strides =  [pow(2,x) for x in reversed(range(nVars))]                    
    result = np.zeros((rows,nVars))
    vals = [0,1] # the values returned by stats.bernoulli
    for col,stride in izip(range(nVars),strides):
        result[:,col] = np.tile(np.repeat(vals,stride),rows/(stride*2))
    return result

def iskip(iterable, skip):
    """ returns an iterator that skips the element at the specified skip index"""
    # iskip('ABCDEFG', 2) --> A B D E F G
    for i, element in enumerate(iterable):
        if i != skip:
            yield element

def epsilon(N,alpha):
    return math.sqrt((1.0/float(2.0*N))*math.log(2.0/alpha))


class BetaPosterior(object):
    def __init__(self,prior = (1,1)):
        self.trueCount = prior[0]
        self.falseCount = prior[1]
        self.N = sum(prior)

    def update(self,result):
        self.N +=1
        if result:
            self.trueCount +=1
        else:
            self.falseCount +=1
            
    def wieghtdUpdate(self,result,weight):
        self.N += weight
        if result:
            self.trueCount+=weight
        else:
            self.falseCount+=weight

    def sample(self):
        dist = beta(self.trueCount,self.falseCount)
        return dist.rvs()

    def  __repr__(self):
        return str((self.trueCount,self.falseCount))
    
    def pOfReward(self):
        return self.trueCount/float(self.trueCount+self.falseCount)

    def upperCI(self,alpha):
        """ get the upper confidence interval on theta"""
        e = epsilon(self.N,alpha)
        return self.pOfReward()+e # I know this is not an actual CI bound as it can exceed 1 (I could take min 1 and this) but maybe I get extra info this way ...


class ProbabilityModel(object):
    
    def do(self,action):
        variable,value = action
        sample = bernoulli.rvs(self.pxlist)  # sample X
        sample[variable] = value # set the intervened variable to its value
        pY = self.pYGivenX(sample) # get P(Y|X)
        y = bernoulli.rvs(pY)  # sample Y from P(Y|X)
        return (y,sample)   
    # want some way of quantifying the variability of pYGivenX

    def getArm(self,action):
        # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        variable,value = action
        return int(variable)*2+int(value)

    def getAction(self,arm):
        # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        value = arm % 2
        variable = arm/2
        return (variable,value)
        
    
    
    
class FullProbabilityModel(ProbabilityModel):
    def __init__(self,pxlist,probYisOne):
        """ probYisOne should be single array of length 2^numCauses.
            The order is assumed to be cycling through the last variable fastest
            ie, eith 3 causes should correspond to [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]"""
        self.numCauses = len(pxlist)
        self.pxlist = np.array(pxlist)
        combinations = generateCombinations(self.numCauses)
        self.yGivenX = dict(zip([tuple(c) for c in combinations ],probYisOne))

        # now calculate marginals for each arm, P(y|x1=0),P(y|x1=1),P(y|x2=0),P(y|x2=1),...
        self.py = [0]*2*self.numCauses
        for index,row in enumerate(combinations):
            p = [self.pxlist[i] if val == 1 else 1-self.pxlist[i] for i,val in enumerate(row)] # the probabilities for each val in X(independent)
            for i,value in enumerate(row):
                excludeI = list(iskip(p,i))
                dp = np.prod(excludeI)*probYisOne[index]  #product of p excluding index i * probYisOne[index]
                field = self.getArm((i,value))
                self.py[field]+=dp # update correct field
                

    def pYGivenX(self,xsample):
        """ calculate P(y|x1...xN) """
        return self.yGivenX.get(tuple(sample))# get P(Y|X)

    

class LogisticProbabilityModel(ProbabilityModel):
    """ models the case where P(x1...xn) = g(w0+w1*x1+w2*x2+...wn*xn) """
    def __init__(self,pxlist,weights,w0 = 0, m = 1000):
        assert(len(pxlist)==len(weights))
        self.pxlist = np.array(pxlist)
        self.weights = weights
        self.w0=w0
        self.numCauses = len(pxlist)
        self.numArms = 2*self.numCauses
        self.monteCarloMarginalize(m)
    

    def monteCarloMarginalize(self,m):
        """ calculate P(y|x1=0), P(y|x1=1) ..."""
        self.py = [0]*self.numArms
        for k in xrange(self.numArms):
            variable,value = self.getAction(k)
            pyk = 0
            for i in xrange(m):
                sample = bernoulli.rvs(self.pxlist)  # sample x1 ... xn
                sample[variable] = value
                pyk += self.pYGivenX(sample)
            pyk = pyk/float(m)
            self.py[k] = pyk

    def bestArm(self):
        """ figure out which is the best arm and what is probability is """
        # if w[i] is -ive then x[i] = 0, if w[i] is +ive x[i] = 1
        return
        

    def pYGivenX(self,xvals):
        return sigmoid(np.dot(xvals,self.weights)+self.w0)
        # dot product between xvals and weigths through sigmoid function

class CausalBinaryBandit(object):
    def __init__(self,model):
        self.model = model
        self.numCauses = len(model.pxlist)
        self.numArms = self.numCauses*2
        self.reset()
        
    def reset(self):
        self.arms = [BetaPosterior() for x in range(2*self.numCauses)] # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        self.pxEstimate = [BetaPosterior() for x in range(self.numCauses)] # hold and estimate of P(X1),P(X2)...
        self.interventions = 0
        self.rewards = 0
        self.sumExpectedReward = 0


    def thompsonUpdate(self,arm,outcome):
        """ update based on results from intervention"""
        y,other = outcome
        self.rewards+=y
        self.arms[arm].update(y)
        
    
    def causalThompsonUpdate(self,arm,outcome):
        y,other = outcome
        self.rewards += y
        armObject = self.arms[arm]
        armObject.update(y) # update intervened are as normal
        variable,value = self.model.getAction(arm)
        
        # figure out the weight
       
        if value == 1:
              xObsVal = self.pxEstimate[variable].trueCount # number of times we observed X to be 1
              xIsVal = xObsVal+armObject.trueCount # total number of times X has been 1
        else:
            xObsVal = self.pxEstimate[variable].falseCount # number of times we observed X to be 0
            xIsVal = xObsVal+armObject.falseCount # total number of times X has been 0

        w = xObsVal/float(xIsVal)
        
        for var,val in enumerate(other):
            if var != variable:
                self.pxEstimate[var].update(val) # update observed estimates for non-intervened variables
                oArm = self.model.getArm((var,val))
                self.arms[oArm].wieghtdUpdate(y,w) # update other arms based on observed values, weighted according to likelyhood this combination would have occured in observational data 

    def pull(self,arm):
        action = self.model.getAction(arm)
        return self.model.do(action)
    
    def UCBSelectArm(self,alpha):
        """ pick the arm with the highest upper bound on its payoff"""
        upperBounds = [arm.upperCI(alpha) for arm in self.arms]
        self.interventions +=1
        selectedArmIndx = upperBounds.index(max(upperBounds))
        self.sumExpectedReward += self.model.py[selectedArmIndx]
        return selectedArmIndx
               
    def selectArm(self):
        """ Returns the index of the next arm to select """
        sampled_theta = [arm.sample() for arm in self.arms]
        self.interventions+=1
        selectedArmIndx = sampled_theta.index( max(sampled_theta))
        self.sumExpectedReward += self.model.py[selectedArmIndx]
        return selectedArmIndx

    def randomSample(self,n):
        """ sample at random"""
        for i in xrange(n):
            arm = randint(-1,self.numArms-1)
            self.interventions +=1
            outcome = self.pull(arm)
            self.thompsonUpdate(arm,outcome)
        return self.regret()

    def thompsonSample(self,n):
        for i in xrange(n):
            arm = self.selectArm()
            outcome = self.pull(arm)
            self.thompsonUpdate(arm,outcome)
        return self.regret()

    def UCBSample(self,n,alpha):
        for i in xrange(n):
            arm = self.UCBSelectArm(alpha)
            outcome = self.pull(arm)
            self.thompsonUpdate(arm,outcome)
        return self.regret()
    
    def causalThompsonSample(self,n,alpha):
        for i in xrange(n):
            arm = self.UCBSelectArm(alpha)
            outcome = self.pull(arm)
            self.causalThompsonUpdate(arm,outcome)
        return self.regret()

    def regret(self):
        """ optimal expected reward minus sum of expected reward of played arms """
        optimal_reward = max(self.model.py)*self.interventions
        return (optimal_reward - self.sumExpectedReward)/(float(self.interventions))






def compareBandits(trials,interventions,o,px,pyGivenX):
    cr = 0
    tr = 0
    cbr = 0
    numArms = len(px)
    bandit = CausalBinaryBandit(numArms)
    bandit.setProbXandProbYGivenX(px,pyGivenX)
    for experiment in range(trials):
        bandit.reset()
        
        bandit.UCBSample(interventions,0.05)
        regret = bandit.regret()
        cbr += regret
        o.write(str(regret)+",UCB,"+str(interventions)+","+str(numArms)+"\n")

        bandit.reset()

        bandit.causalThompsonSample(interventions)
        regret = bandit.regret()
        cr += regret
        o.write(str(regret)+",CT,"+str(interventions)+","+str(numArms)+"\n")
        
        bandit.reset()
        
        bandit.thompsonSample(interventions)
        regret = bandit.regret()
        tr += regret
        o.write(str(regret)+",T,"+str(interventions)+","+str(numArms)+"\n")
    return (cr/float(trials),tr/float(trials),cbr/float(trials))


# requires structural but not parametric assumptions

# importance sampling technique will extend to more complex combinations of variables, provided do(xi) is identifible for at least some arms (for those it isn't we get no improvement) (I'm pretty sure)

# what about if we are allowed to intervene on multiple arms simultaneously?




##o = open("thompsonvsUCBvsCT2.txt","w")
##px = [0.5,0.5,0.5]
##pyGivenX = [0.1,0.3,0.2,0.5,.05,.03,.2,.3]
##for ni in [pow(2,x) for x in range(2,12)]:
##    print compareBandits(1000,ni,o,px,pyGivenX)
##o.close()


# add additive noise model


##o = open("causal_thompson.txt","w")
##for numArms in [pow(2,x) for x in range(1,4)]:
##    print numArms
##    trials = 5
##    px = [0.5]*numArms
##    pyGivenX = [0.5]*pow(2,numArms)
##    pyGivenX[0:pow(2,numArms-1)] = [0.6]*pow(2,numArms-1) # think further about this step ...
##    interventions = 10
##    print compareBandits(trials,interventions,o,px,pyGivenX)
##    
##o.close()

# gets a lot slower as the number of arms increases - not entirely obvious why - would be good to know which method spent most time running - is it selectArm?
# actually appears to be in intitialization - the marginalization step gets expensive as graph size increases - one option would be to allow specification of marginal probabilities
# and select joint probabilities such that marginals known analytically. 
            

    
