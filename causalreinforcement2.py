from libcause import BinaryNetworkBandit
from numpy import *
from scipy.stats import *


# A second attempt at causal reinforcement learning
# restricted to the simpler case where we can only intervene on one variable at a time

interventions = [{"X1":'0'},{"X1":'1'},{"X2":'0'},{"X2":'1'},{"X3":'0'},{"X3":'1'}] 
bandit = BinaryNetworkBandit("bayesnet.json",interventions,'Y')

simulations = 1
samples = 12
best_arm = 0.4
o = open("causalreinforcementresults2.txt","w")

for i in xrange(simulations):
    bandit.sample(samples)
    regret = bandit.regret(best_arm)
    o.write("Thompson,"+str(regret)+"\n")
    bandit.reset()

    bandit.sample_with_network_info(10,["X1","X2","X3"])
    regret = bandit.regret(best_arm)
    o.write("CThompson,"+str(regret)+"\n") 
    bandit.reset()
    
o.close()



print "DONE"
