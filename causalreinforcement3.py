from multicausesim import *
import numpy as np
from scipy.stats import bernoulli 
import matplotlib.pyplot as plt

o = open("causalreinforce4.txt","w")
experiments = 500
tslist = [100,500]
for timesteps in tslist:
    for k in [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,150,200]:
        print "k",k
        w = np.zeros(k)
        w[0] = 1
        for i in xrange(experiments): # make this many models for each arm
            model = LogisticProbabilityModel([0.5]*k,w,w0=-1,m=10000)
            bandit = CausalBinaryBandit(model)
            ucbRegret = bandit.UCBSample(timesteps,0.05)
            result = ",".join([str(x) for x in [k,i,ucbRegret,timesteps,"ucb"]])+"\n"
            o.write(result)
            bandit.reset()
            causalRegret = bandit.causalThompsonSample(timesteps,0.05)
            result = ",".join([str(x) for x in [k,i,causalRegret,timesteps,"causal"]])+"\n"
            o.write(result)
        o.flush()
        
        
    
    #print model.py


# mean goes up and variance goes down as num arms goes up - in current model...
