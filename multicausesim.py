import numpy as np
from scipy.stats import bernoulli
from scipy.stats import beta
from itertools import izip
from random import randint

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
       


class BetaPosterior(object):
    def __init__(self,prior = (1,1)):
        self.trueCount = prior[0]
        self.falseCount = prior[1]

    def update(self,result):
        if result:
            self.trueCount +=1
        else:
            self.falseCount +=1
            
    def wieghtdUpdate(self,result,weight):
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
        
        

class CausalBinaryBandit(object):
    def __init__(self,numCauses):
        self.numCauses = numCauses
        self.setProbXandProbYGivenX(beta.rvs(1,1,size=numCauses),beta.rvs(1,1,size=pow(2,numCauses)))
        self.arms = [BetaPosterior() for x in range(2*numCauses)] # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        self.numArms = numCauses*2
        self.pxEstimate = [BetaPosterior() for x in range(numCauses)] # hold and estimate of P(X1),P(X2)...
        self.interventions = 0
        self.rewards = 0
        
        
    def setProbXandProbYGivenX(self,pxlist,probYisOne):
        """ probYisOne should be single array of length 2^numCauses.
            The order is assumed to be cycling through the last variable fastest
            ie, eith 3 causes should correspond to [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]"""
        assert(len(pxlist)==self.numCauses)
        self.pxlist = np.array(pxlist)
        combinations = generateCombinations(self.numCauses)
        self.yGivenX = dict(zip([tuple(c) for c in combinations ],probYisOne))

        # now calculate marginals for each arm, P(y|x1=0),P(y|x1=1),P(y|x2=0),P(y|x2=1),...
        self.py = [0]*2*self.numCauses
        for index,row in enumerate(combinations):
            p = [self.pxlist[i] if val == 1 else 1-self.pxlist[i] for i,val in enumerate(row)] # the probabilities for each val in X(independent)
            for i,value in enumerate(row):
                dp = np.prod(list(iskip(p,i)))*probYisOne[index]  #product of p excluding index i * probYisOne[index]
                field = self.getArm((i,value))
                self.py[field]+=dp # update correct field
                
                        
    def do(self,arm):
        """perform the intervention, set the variable variable to value. Returns a the values of all the causes and of Y"""
        action = self.getAction(arm)
        variable,value = action
        sample = bernoulli.rvs(self.pxlist)  # sample X
        sample[variable] = value # set the intervened variable to its value
        pY = self.yGivenX.get(tuple(sample))# get P(Y|X)
        y = bernoulli.rvs(pY)  # sample Y from P(Y|X)
        return (y,sample)

    def getArm(self,action):
        # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        variable,value = action
        return int(variable)*2+int(value)
        
    def getAction(self,arm):
        # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        value = arm % 2
        variable = arm/2
        return (variable,value)

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
        variable,value = self.getAction(arm)
        
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
                oArm = self.getArm((var,val))
                self.arms[oArm].wieghtdUpdate(y,w) # update other arms based on observed values, weighted according to likelyhood this combination would have occured in observational data 
            
        
        
        
    def selectArm(self):
        """ Returns the index of the next arm to select """
        sampled_theta = []
        for indx,arm in enumerate(self.arms):
            sampled_theta.append(arm.sample())
        self.interventions+=1
        return sampled_theta.index( max(sampled_theta))

    def randomSample(self,n):
        """ sample at random"""
        for i in xrange(n):
            arm = randint(-1,self.numArms-1)
            self.interventions +=1
            outcome = self.do(arm)
            self.thompsonUpdate(arm,outcome)

    def thompsonSample(self,n):
        for i in xrange(n):
            arm = self.selectArm()
            outcome = self.do(arm)
            self.thompsonUpdate(arm,outcome)

    def causalThompsonSample(self,n):
        for i in xrange(n):
            arm = self.selectArm()
            outcome = self.do(arm)
            self.causalThompsonUpdate(arm,outcome)
            

    def regret(self):
        """ total reward vs expected reward had optimal arm been played from the start """
        optimal_reward = max(self.py)*self.interventions
        return (optimal_reward - self.rewards)/(float(self.interventions))

o = open("causal_thompson.txt","w")
cr = 0
tr = 0
trials = 100
interventions = 15
for experiment in range(trials):
    print experiment
    bandit = CausalBinaryBandit(4)
    bandit.setProbXandProbYGivenX([0.2,0.1,0.5,.3],[0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8,.3,.01,.2,.9,.7,.2,.5,.005])
    bandit.causalThompsonSample(interventions)
    regret = bandit.regret()
    cr += regret
    o.write(str(regret)+",CT\n")
    
    bandit = CausalBinaryBandit(4)
    bandit.setProbXandProbYGivenX([0.2,0.1,0.5,.3],[0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8,.3,.01,.2,.9,.7,.2,.5,.005])
    bandit.thompsonSample(interventions)
    regret = bandit.regret()
    tr += regret
    o.write(str(regret)+",T\n")
o.close()
print "CAUSAL",cr/float(trials)
print "STANDARD",tr/float(trials)
            
    
    


    
