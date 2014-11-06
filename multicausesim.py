import numpy as np
from scipy.stats import bernoulli
from scipy.stats import beta
from itertools import izip
from random import randint
import math
from matplotlib import *
import matplotlib.pyplot as plt


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

def getArm(action):
    # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
    variable,value = action
    return int(variable)*2+int(value)

def getAction(arm):
    # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
    value = arm % 2
    variable = arm/2
    return (variable,value)

def flip(binary):
    """ turns 0 -> 1 and 1 -> 0 """
    return binary*-1+1




class Arm(object):
    def __init__(self,numArms,armIndx,alpha = 0.05):
        self.counts = [[1,1] for x in xrange(numArms)] # first entry is trials, second is success
        self.id = armIndx
        self.action = getAction(armIndx) # tuple (variable,value) that is the action corresponding to this arm
        self.alpha = alpha
        # One of the other arms also corresponds to the same variable and should not be updated
        self.sisterID = getArm((self.action[0],flip(self.action[1]))) 
        
    def update_intervention(self,result):
        """ updates the arm based on the result of pulling that arm"""
        self.counts[self.id][0]+=1
        self.counts[self.id][1]+= result

    def update_observation(self,otherArmID,result):
        """ updates this arm based on the result of pulling otherArm"""
        self.counts[otherArmID][0]+=1
        self.counts[otherArmID][1]+=result

    def prYdisabled(self, pxlist): # pxlist holds the probability each variable is 1
        """ returns an estimation of the probability of getting a reward by pulling this arm and an uncertainty"""
        # each other arm corresponds to a key value pair ie (x1,0), (x1,1),(x3,0),(x3,1)
        
        pr = 0
        totalW = 0
        varID = 0
        for i in range(0,len(self.counts),2):
            if varID != self.action[0]: #
                
                # combine i and i+1 - these correspond to parts A and B
                cA = self.counts[i]
                cB = self.counts[i+1]
                A = cA[1]/float(cA[0])
                B = cB[1]/float(cB[0])
                eA = epsilon(cA[0],self.alpha)
                eB = epsilon(cB[0],self.alpha)
                pXIs1 = pxlist[varID]
                theta = A*(1-pXIs1)+B*pXIs1
                etheta =  eA*(1-pXIs1)+eB*pXIs1 # a dodgy standard deviation in the estiate of theta
            else:
                # add in the ones from actual interventions on this variable.
                cS = self.counts[self.id]
                theta = cS[1]/float(cS[0])
                etheta = epsilon(cS[0],self.alpha)
            w = 1/pow(etheta,2)
            pr+=theta*w
            totalW +=w
            varID +=1

        
        pr = pr/totalW
        er = pow(1.0/totalW,2)
        return (pr,er)

    def prY(self,pxlist,beta):
        pr = 0
        totalW = 0
        varID = 0
        for i in range(0,len(self.counts),2): # stepping throught the variables
            if varID != self.action[0]:
                assert(i!=self.id and i!=self.sisterID)

                na0 = float(self.counts[i][0])
                ma0 = self.counts[i][1]
                na1 = float(self.counts[i+1][0])
                ma1 = self.counts[i+1][1]
                p = pxlist[varID]

                n = .5*min(na1/p,na0/(1-p))
                pr += n*(p*ma1/na1+(1-p)*ma0/na0)
                totalW += n
                    
            else:
                # add in the ones from actual interventions on this variable.
                mij = self.counts[self.id][1]
                nij = float(self.counts[self.id][0])
                    
                n = nij
                pr += n*mij/nij
                totalW += n
     
            varID +=1

        pr = pr/totalW
        er = math.sqrt((beta/totalW)*math.log(1/self.alpha))    
        return (pr,er)
                
    def upperCI(self,armProbs,beta):
        """ returns the upper confidence bound on prY """
        pr,er = self.prY(armProbs,beta)
        return pr+er

    


class CausalBandit2(object):
    def __init__(self,model,beta):
        self.model = model
        self.numCauses = len(model.pxlist)
        self.numArms = self.numCauses*2
        self.beta = beta
        self.reset()
        
    def reset(self):
        self.arms = [Arm(self.numArms,i) for i in range(self.numArms)] # arms are ordered as X1=0,X1=1,X2=0,X2=1,...
        self.interventions = 0
        self.rewards = 0
        self.sumExpectedReward = 0
        
    
    def update(self,armID,outcome):
        """ called when armID was pulled with given outcome """
        y,other = outcome
        self.rewards += y
        arm = self.arms[armID]
        arm.update_intervention(y)
    
        for var,val in enumerate(other):
            if var != arm.action[0]:
                oArm = self.arms[getArm((var,val))]
                oArm.update_observation(armID,y)
                
    def pull(self,arm):
        action = getAction(arm)
        return self.model.do(action)
    
    def selectArm(self,alpha):
        """ pick the arm with the highest upper bound on its payoff"""
        upperBounds = [arm.upperCI(self.model.pxlist,self.beta) for arm in self.arms]
        self.interventions +=1
        selectedArmIndx = upperBounds.index(max(upperBounds))
        self.sumExpectedReward += self.model.py[selectedArmIndx]
        return selectedArmIndx
               
    
    def sample(self,n,alpha,verbose=False):
        for i in xrange(n):
            arm = self.selectArm(alpha)
            outcome = self.pull(arm)
            self.update(arm,outcome)
            if verbose:
                print "arm:"+str(arm)+"="+str(getAction(arm))+"->"+str(outcome)
                print [arm.prY(self.model.pxlist,self.beta)[0] for arm in self.arms]
        return self.regret()

    def regret(self):
        """ optimal expected reward minus sum of expected reward of played arms """
        optimal_reward = max(self.model.py)*self.interventions
        return (optimal_reward - self.sumExpectedReward)/(float(self.interventions))





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

    def plot(self,axis):
        if axis == None:
            f,axis = subplots()
        dist = beta(self.trueCount,self.falseCount)
        x = np.linspace(0,1,100)
        axis.plot(x,dist.pdf(x))
        return axis
        

class ProbabilityModel(object):
    
    def do(self,action):
        variable,value = action
        sample = bernoulli.rvs(self.pxlist)  # sample X
        sample[variable] = value # set the intervened variable to its value
        pY = self.pYGivenX(sample) # get P(Y|X)
        y = bernoulli.rvs(pY)  # sample Y from P(Y|X)
        return (y,sample)   
   
    
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
                field = getArm((i,value))
                self.py[field]+=dp # update correct field
                

    def pYGivenX(self,sample):
        """ calculate P(y|x1...xN) """
        return self.yGivenX.get(tuple(sample))# get P(Y|X)



class TrivialProbabilityModel(ProbabilityModel):
    """ p(x1)...p(xn) = 0.5 and y independent of all x except x1. P(y|x1=1) = 0.6, P(y|x1=0) = 0.5 """
    def __init__(self,numArms,epsilon):
        assert(numArms % 2 == 0)
        self.epsilon = epsilon
        self.py = [.5*(1+self.epsilon)]*numArms
        self.py[0] = 0.5
        self.py[1] = 0.5+self.epsilon
        self.pxlist = np.array([0.5]*(numArms/2))

    def pYGivenX(self,xvals):
        return 0.5+self.epsilon*xvals[0]

    
        

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
        action = getAction(arm)
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

def plot(self,axis):
        if axis == None:
            f,axis = subplots()
        dist = beta(self.trueCount,self.falseCount)
        x = np.linspace(0,1,100)
        axis.plot(x,dist.pdf(x))
        return axis

def tryBeta(betalst):
    experiments = 100
    n = 100
    model = FullProbabilityModel([0.5,0.1],[0.1,0.3,0.5,0.2])
    means = []
    stds = []
    for beta in betalst:
        data = np.zeros((experiments,1))

        for i in xrange(experiments):
            bandit2 = CausalBandit2(model,beta)
            bandit2.sample(n,0.05,verbose=False)
            data[i,0] = bandit2.regret()
        means.append(np.mean(data))
        stds.append(np.std(data))
    f,axis = plt.subplots()
    ste = np.array(stds)/math.sqrt(experiments)
    axis.errorbar(betalst,means,yerr=ste,fmt="o",label="ucb")
    axis.set_xlabel("beta")
    axis.set_ylabel("regret")
    axis.set_title("regret vs beta")
    #plt.show()
    plt.savefig("regretvsbeta.png")
    plt.savefig("regretvsbeta.pdf")


beta = 0.6
n = 1000
experiments = 100
cmeans = []
cstds = []
means = []
stds = []
o = open("manyarmedbandit.txt","w")
armslst = [pow(2,x) for x in range(2,11)]
for numArms in armslst:
    print numArms,"arms",
    model = TrivialProbabilityModel(numArms,0.3)

    data = np.zeros((experiments,2))
    for i in xrange(experiments):
        bandit = CausalBinaryBandit(model)
        bandit.UCBSample(n,0.05)
        bandit2 = CausalBandit2(model,beta)
        bandit2.sample(n,0.05)
        data[i,0] = bandit2.regret()
        data[i,1] = bandit.regret()
        if i % 5 == 0:
            print i,
    print ""
        
    
    m = np.mean(data,axis=0)
    sd = np.std(data,axis=0)/math.sqrt(experiments)
    l = list(m)
    s = list(sd)
    l.extend(s)
    l.append(numArms)
    l = [str(x) for x in l]
    o.write(",".join(l)+"\n")
    o.flush()
    cmeans.append(m[0])
    cstds.append(sd[0])
    means.append(m[1])
    stds.append(sd[1])

o.close()

f,axis = plt.subplots()
axis.errorbar(armslst,means,yerr=stds,fmt="o",label="ucb")
axis.errorbar(armslst,cmeans,yerr=cstds,fmt="D",label="cucb")
axis.set_xlabel("number of arms")
axis.set_ylabel("regret")
axis.set_title("regret vs number of arms")
axis.legend(loc="lower right",numpoints=1)
plt.savefig("regretvsarms.png")
plt.savefig("regretvsarms.pdf")


plt.show()

    


print "DONE"

##experiments = 100
##for i in xrange(experiments):
##    pxlist = np.random.uniform(0,1,size=2)
##    pygivenx = np.random.uniform(0,1,size=4)
##    model = FullProbabilityModel(pxlist,[0.1,0.3,0.5,0.2])
##    bandit = CausalBandit2(model)
##    bandit2 = CausalBinaryBandit(model)
##    for n in [10,50,100,200,500,1000]:
##        
##        bandit.sample(n,0.05)
##        bandit2.UCBSample(n,0.05)
##        data[i,0] = bandit.regret()
##        data[i,1] = bandit2.regret() #update and see what happens to regret with n for each type of bandit.Repeat for increasing number of arms.
##
###print data
##print np.mean(data,axis=0)
##print np.std(data,axis=0)/math.sqrt(experiments)

#print "actual",model.py
#print "causal bandit",[arm.prY(pxlist)[0] for arm in bandit.arms]
#print "normal bandit",[arm.pOfReward() for arm in bandit2.arms]





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
            

    
