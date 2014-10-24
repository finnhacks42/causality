from numpy import *
from scipy.stats import *
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
import itertools
import matplotlib.pyplot as plt
import time
        

class BinaryNetworkBandit(object):
    """Class represents a Discrete Bayesian Network that supports Thompson sampling.
        Currently only supports interventions on a single variable at a time"""
    def __init__(self, bayesnetfile, targetVar = 'Y', prior=(1.0,1.0)):
        self.file = bayesnetfile
        self.theta_range = linspace(0.0001,0.9999,100)
        self.set_bayesnet()
        self.y = targetVar
        self.reset(prior = prior)

    def getCPD(self):
        """ required as querying a TableCPDFactorization leads modifies the underlying Bayesian network (refresh() appears broken) """
        self.set_bayesnet()
        return TableCPDFactorization(self.bn)

    def set_bayesnet(self):
        nd = NodeData()
        skel = GraphSkeleton()
        nd.load(self.file)
        skel.load(self.file)
        skel.toporder()
        self.bn = DiscreteBayesianNetwork(skel, nd)
        
    def reset(self,prior=(1.0,1.0)):
        """ Clears all the data on samples that have been taken so far but keeps graph structure.
            You can optionally specify a new prior"""  
        # possible single interventions on non-target variable
        self.prior = prior
        self.interventions = [] # a list of possible assignments
        self.variables = [] # store the variables - defines an ordering
        values = []
        for variable, data in self.bn.Vdata.iteritems():
            if variable != self.y:
                self.variables.append(variable)
                vals = data.get("vals")
                values.append(vals)
                for value in vals:
                    self.interventions.append({variable:value})

        # lets calculate and print the actual value of theta for each arm (since we know it)
        
        truth = []
        for i in self.interventions:
             cpd = self.getCPD()
             answer = cpd.specificquery({self.y:"1"},i)
             truth.append(answer) 
        print "THETA",truth
        cpd = self.getCPD() # reset the network to its original state
                    

        # generate all possible assignments (intervention on all variables) to non-target values
        combinations = list(itertools.product(*values))
   
        self.assignements = [dict(zip(self.variables,v)) for v in combinations]
        num_assignments = len(self.assignements)
        self.assingment_map = dict(zip([str(list(v)) for v in combinations],range(num_assignments))) # builds a map from a each assingment to its indx
        
        self.atrials =  self.trials = zeros(shape=(num_assignments,), dtype=int) # stores how often each assignment occured
        self.asuccesses = zeros(shape=(num_assignments,), dtype=int) # stores how often each assignment paid off

        self.num_arms = len(self.interventions)
        self.trials = zeros(shape=(self.num_arms,), dtype=int) # stores how often each arm was selected
        self.successes = zeros(shape=(self.num_arms,), dtype=int) # stores how often each arm paid off

        # now here I'm going to assume models where X1 ... Xn mutually independent causes of Y

        # record distributions for P(X1), P(X2) ... - they update only when we observe Xn not when we do it
        self.observed_trials = zeros(shape=(self.num_arms,), dtype=int)
        self.observed_true = zeros(shape=(self.num_arms,), dtype=int)
                
    
    def sample(self,n,plot=-1):
        """ returns n samples based on Thompson sampling """
        for i in xrange(n):
            if plot > 0 and i % plot == 0:
                do_plot = True
            else:
                do_plot = False
            arm = self.get_recommendation(do_plot)
            intervention = self.interventions[arm]
            # note evidence is equivelent to do in libpgm
            result = self.bn.randomsample(1,evidence=intervention)[0] # returns a dictionary containing values for each variable
            reward = int(result.get(self.y))
           
            # update the counts for the pulled arm (P(Y|X?=?))
            self.trials[arm] = self.trials[arm] + 1
            if (reward == 1):
                self.successes[arm] = self.successes[arm] + 1

            
            # for all variables we did not intervene on, update observed
            values = []
            for indx,v in enumerate(self.variables):
                value = result[v]
                values.append(value)
                if v not in intervention:
                    self.observed_trials[indx] += 1
                    if int(value) == 1:
                        self.observed_true[indx]+=1
            
            # update relevent exact assignment
            key = str((values))
            a = self.assingment_map[key]
            self.atrials[a] = self.atrials[a]+1
            if (reward == 1):
                self.asuccesses[a] = self.asuccesses[a] + 1

    def plot_observed(self):
        # put labels under each plot
        f,sp = plt.subplots(1,len(self.variables),sharey=False,figsize=(15,5))
        for i in range(len(self.variables)):
            dist = beta(self.observed_true[i], self.observed_trials[i]-self.observed_true[i])
            sp[i].plot(self.theta_range,dist.pdf(self.theta_range))
        plt.show()

    def plot_assignments(self):
        f,sp = plt.subplots(1,len(self.assingment_map),sharey=False,figsize=(15,5))
        for i in range(len(self.assingment_map)):
            dist = beta(self.asuccesses[i], self.atrials[i]-self.asuccesses[i])
            sp[i].plot(self.theta_range,dist.pdf(self.theta_range))
        plt.show()
        
 
    def get_recommendation(self,do_plot=False):
        """ recommends which arm to pull next proportional to the estimated probability that it is the optimal one"""
        sampled_theta = []

        if do_plot:
            f,sp = plt.subplots(1,self.num_arms,sharey=False,figsize=(15,5))
        for i in range(self.num_arms):
            #Construct beta distribution for posterior
            dist = beta(self.prior[0]+self.successes[i], self.prior[1]+self.trials[i]-self.successes[i])

            if do_plot:
                sp[i].plot(self.theta_range,dist.pdf(self.theta_range))
            
            #Draw sample from beta distribution
            sampled_theta.append(dist.rvs())

            # Alternately calculate P(Y|X1) as sum(P(Y|X1,X2)P(X2)
            # Do this here ....
            
        if do_plot:
            plt.show()
      
        # Return the index of the sample with the largest value
        return sampled_theta.index( max(sampled_theta) )
    
    
             
    def regret(self, bestprob):
        """ regret as ratio between reward and expectation of reward had we always selected best """
        reward = sum(self.successes)/float(sum(self.trials))
        optimal = bestprob
        return 1 - reward/bestprob


bandit = BinaryNetworkBandit("bayesnet.json")
bandit.sample(500,plot=500)
print bandit.trials
print bandit.successes
bandit.plot_observed()
bandit.plot_assignments()