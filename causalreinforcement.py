from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.sampleaggregator import SampleAggregator
from scipy.stats import *
import json
from numpy import *


class NetworkBetaBandit(object):
    def __init__(self, bayesnet, interventions, targetVar, prior=(1.0,1.0)):
        num_options = len(interventions)
        self.trials = zeros(shape=(num_options,), dtype=int)
        self.successes = zeros(shape=(num_options,), dtype=int)
        self.num_options = num_options
        self.prior = prior
        self.bn = bayesnet
        self.interventions = interventions
        self.y = targetVar
        
    def sample(self,n):
        for i in xrange(n):
            arm = self.get_recommendation()
            # note evidence is equivelent to do in libpgm
            result = bn.randomsample(1,evidence=interventions[arm])[0]
            reward = result.get(self.y)
            bandit.add_result(arm,reward)
 
    def add_result(self, trial_id, success):
        self.trials[trial_id] = self.trials[trial_id] + 1
        if (success):
            self.successes[trial_id] = self.successes[trial_id] + 1
 
    def get_recommendation(self):
        sampled_theta = []
        for i in range(self.num_options):
            #Construct beta distribution for posterior
            dist = beta(self.prior[0]+self.successes[i],
                        self.prior[1]+self.trials[i]-self.successes[i])
            #Draw sample from beta distribution
            sampled_theta += [ dist.rvs() ]
        # Return the index of the sample with the largest value
        return sampled_theta.index( max(sampled_theta) )
    
    def regret(self, bestprob):
        # regret as ratio between reward and expectation of reward had we always selected best
        reward = sum(self.successes)/float(sum(self.trials))
        optimal = bestprob
        return 1 - reward/bestprob

# load nodedata and graphskeleton
nd = NodeData()
skel = GraphSkeleton()
nd.load("bayesnet.json")    # any input file
skel.load("bayesnet.json")

# topologically order graphskeleton
skel.toporder()

# load bayesian network
bn = DiscreteBayesianNetwork(skel, nd)

simulations = 10000 # the number of simulations of the whole process
experiments = 32 # the number of experiments we run in each simulation

# specify what the interventions are for the 'try all combinations of interventions' bandit
interventions = [{"X1":'0',"X2":'0',"X3":"0"},{"X1":'0',"X2":'0',"X3":"1"},{"X1":'0',"X2":'1',"X3":"0"},{"X1":'0',"X2":'1',"X3":"1"},{"X1":'1',"X2":'0',"X3":"0"},{"X1":'1',"X2":'0',"X3":"1"},{"X1":'1',"X2":'1',"X3":"0"},{"X1":'1',"X2":'1',"X3":"1"}]

regrets = []
progress = 0
for i in range(simulations):
    if (100*i)/simulations != progress:
        progress +=1
        print "part1",progress,"%"
    bandit = NetworkBetaBandit(bn,interventions,'Y')
    bandit.sample(experiments)
    regrets.append(bandit.regret(.4))

# a rough estimate how well we do with Bayes Bandits, where we have only done a limited number of experiments

#print np.mean(regrets),"+-", sem(regrets)

# If we know the causal model is X1 -> Y <- X2, then P(Y|do(X1)) = P(Y|X1)
# 

#hmmm this doesn't get to do final optimization as ultimately can't narrow down enough - It should be setting each arm ... eventually. Ie it should learn ...
interventions = [{"X1":'0'},{"X1":'1'},{"X2":'0'},{"X2":'1'},{"X3":'0'},{"X3":'1'}] 
X1arms = {0:0,1:1}
X2arms = {0:2,1:3}
X3arms = {0:4,1:5}

regrets2 = []
progress = 0
for i in range(simulations):
    if (100*i)/simulations != progress:
        progress +=1
        print "part2",progress,"%"
    bandit = NetworkBetaBandit(bn,interventions,'Y')
    for j in xrange(experiments):
        arm = bandit.get_recommendation()
        result = bn.randomsample(1,evidence=bandit.interventions[arm])[0]
       
        # add results for two arms, one for what X1 was and one for what X2 was
        a1 = X1arms.get(int(result.get("X1")))
        a2 = X2arms.get(int(result.get("X2")))
        a3 = X3arms.get(int(result.get("X3")))
        reward = result.get('Y')
        bandit.add_result(a1,reward)
        bandit.add_result(a2,reward)
        bandit.add_result(a3,reward)
        regrets2.append(bandit.regret(.4))

# a rough estimate how well we do with Causal Bayes Bandit - appears empirically better even in this setting. Now verify theoretically
#print np.mean(regrets2),"+-", stats.sem(regrets2)
o = open("causalreinforceresults.txt","w")
for i in xrange(simulations):
    o.write(str(regrets[i])+","+str(regrets2[i])+"\n")
o.close()



