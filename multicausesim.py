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
        self.numArms = numCauses*2
        self.reset()
        
    def reset(self):
        self.arms = [BetaPosterior() for x in range(2*self.numCauses)] # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        self.pxEstimate = [BetaPosterior() for x in range(self.numCauses)] # hold and estimate of P(X1),P(X2)...
        self.interventions = 0
        self.rewards = 0
        self.sumExpectedReward = 0
        
    
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
                excludeI = list(iskip(p,i))
                dp = np.prod(excludeI)*probYisOne[index]  #product of p excluding index i * probYisOne[index]
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
        selectedArmIndx = sampled_theta.index( max(sampled_theta))
        self.sumExpectedReward += self.py[selectedArmIndx]
        return selectedArmIndx

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
        """ optimal expected reward minus sum of expected reward of played arms """
        optimal_reward = max(self.py)*self.interventions
        return (optimal_reward - self.sumExpectedReward)/(float(self.interventions))



def compareBandits(trials,interventions,o,px,pyGivenX):
    cr = 0
    tr = 0
    numArms = len(px)
    bandit = CausalBinaryBandit(numArms)
    bandit.setProbXandProbYGivenX(px,pyGivenX)
    for experiment in range(trials):
        print experiment
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
    return (cr/float(trials),tr/float(trials))

o = open("causal_thompson.txt","w")
for numArms in [pow(2,x) for x in range(1,5)]:
    print numArms
    trials = 5
    px = [0.5]*numArms
    pyGivenX = [0.5]*pow(2,numArms)
    pyGivenX[0:pow(2,numArms-1)] = [0.6]*pow(2,numArms-1) # think further about this step ...
    interventions = 10
    print compareBandits(trials,interventions,o,px,pyGivenX)
    
o.close()

# gets a lot slower as the number of arms increases - not entirely obvious why - would be good to know which method spent most time running - is it selectArm?
# actually appears to be in intitialization - the marginalization step gets expensive as graph size increases - one option would be to allow specification of marginal probabilities
# and select joint probabilities such that marginals known analytically. 
            


    
