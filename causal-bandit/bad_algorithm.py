from math import *
import random

def randb(p):
    """ returns 1 with probability p and 0 with probability (1-p)"""
    r = random.random()
    return int(r < p)

class Estimate(object):
    def __init__(self,delta):
        self.m = 1
        self.n = 1
        self.delta = delta
        self.w = w
    def update(self,value):
        self.n+=1
        self.m+=value
    def get(self):
        return self.m/float(self.n)

    def count(self):
        return self.n

    def e(self):
        return sqrt((1/(2.0*self.n))*log(2.0/self.delta))


def n_a(v1,v0,p):
    return .5*min(v1.count()/float(p),v0.count()/float(1-p))

def weights(v21,v20,v31,v30,p2,p3):
    na2 = n_a(v21,v20,p2)
    na3 = n_a(v31,v30,p3)
    t = float(na2+na3)
    return (na2/t,na3/t)

def algorithm(v21,v20,v31,v30,p2,p3):
    """ runs the algorithm for a given step, chosing some action to take and doing updates. """
    #### insert action selection to make it as biased as possible
    

    ####

    v2 = v21.get()*p+v20.get()*(1-p)
    v3 = v31.get()*p+v30.get()*(1-p)
    w2,w3 = weights(v21,v20,v31,v30,p2,p3)
    v = w2*v2+w3*v3
    return v


# We can calculate the expectation of v2, and v3 after t timesteps (for a given algorithm)
# We can calculate how far the true estimate is from the actual one. 


delta = .05
p2 = 0.5
p3 = 0.5
v21 = Estimate(delta)
v20 = Estimate(delta)
v31 = Estimate(delta)
v30 = Estimate(delta)



 

n = 1000
for i in range(n):
    v = randb(.7)
    v11.update(v)

print v11.get()
print v11.e()
        

