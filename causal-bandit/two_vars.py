# lets run a simulation to see how we should allocate our samples
# to get within epsilon bounds for all arms as quick as possible
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def split_samples(n,p1,p2):
    """ return the number of times to take each action, subject to a total of n"""
    equal = .25*n
    return (equal,equal,equal,equal) # do(x1=1), do(x1=0), do(x2=1), do(x2=0)

def epsilon(a):
    return np.sqrt(np.log(2/.5)/(2.0*a))

def a(n,na1,na0,pa):
    nom = na1*na0
    den = na1*pow(1-pa,2)+na0*pow(pa,2)
    try:
        assert((den > 0.0).all())
    except:
        print min(den)
    return n+nom/den

def simulate(split,p1,p2,runs):
    n11,n10,n21,n20 = split
    n11_21 = binom.rvs(n11, p2, size=runs)
    n11_20 = (n11 - n11_21)
    n10_21 = binom.rvs(n10,p2,size=runs)
    n10_20 = (n10-n10_21)
    n21_11 = binom.rvs(n21,p1,size=runs)
    n21_10 = (n21- n21_11)
    n20_11 = binom.rvs(n20,p1,size=runs)
    n20_10 = (n20- n20_11)

    # deal with the fact some of these n's can be 0.

    a11 = a(n11,n21_11,n20_11,p2)
    a10 = a(n10,n21_10,n20_10,p2)
    a21 = a(n21,n11_21,n10_21,p1)
    a20 = a(n20,n11_20,n10_20,p1)
   

    e11 = epsilon(a11)
    e10 = epsilon(a10)
    e21 = epsilon(a21)
    e20 = epsilon(a20)

    e = [np.mean(e11),np.mean(e10),np.mean(e21),np.mean(e20)]
    worst = max(e)
    return worst

    #plt.hist(a11)
    #plt.show()
    
    
n = 100.0 # total number of samples
p1 = .5 # P(X1=1)
p2 = .5 # P(X2=1)


results = []
for f1 in np.linspace(.1,.9,9):
    for f11 in np.linspace(.1,.9,9):
        n11 = n*f1*f11
        n10 = n*f1*(1-f11)
        for f21 in np.linspace(.1,.9,9):
            n21 = n*(1-f1)*f21
            n20 = n*(1-f1)*(1-f21)
            assert(abs(n11+n10+n21+n20-n) < .000001)
            split = (n11,n10,n21,n20)
            print split
            time.sleep(1)
            largest_epsilon = simulate(split,p1,p2,100)
            results.append((split,largest_epsilon))


print min(results,key=lambda x: x[1])




