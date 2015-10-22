from numpy import *
from scipy.stats import binom
import matplotlib.pyplot as plt
from scipy import stats

def calc_m(array): # assumes array already sorted smallest to largest
    for i,v in enumerate(array):
        if v >= 1/(i+1):
            return i+1
    print("returning len")
    return len(array)

def simulate_estimation_of_q_and_m(N,m,delta,simulations):
    # simulate observering for h rounds and estimating bar(q_i)
    q = zeros(N)
    q[m-1:] = 1/m
    h = 12*m*log(4*N/delta)
    print("h",h)
    est_m = zeros(simulations)
    for i in range(simulations):
        counts = binom.rvs(h,q) #For each q_i the the count will be ~ binomial(h,q_i)
        est_q = counts/h
        est_bar_q = sort(minimum(est_q,1-est_q)) #the algorithm doesn't know q_i < .5
        est_m[i] = calc_m(est_bar_q)

    freq = stats.itemfreq(est_m)
    print(freq)
    fig,ax = plt.subplots(1,1)
    plt.bar(freq[:,0],freq[:,1],align='center')
    plt.show()

def m_versus_q(Nvals,simulations):
    results = zeros((simulations, len(Nvals)))
    for i,s in enumerate(range(simulations)):
        for j,N in enumerate(Nvals):
            q = random.uniform(high=0.5,size = N)
            q = sort(q)
            m = calc_m(q)
            results[i,j] = m
    f, ax = plt.subplots(1, 1)
    error = results.std(axis=0)
    ax.errorbar(Nvals, results.mean(axis=0), yerr=error, fmt='o')
    #ax.plot(Nvals,results.mean(axis=0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    return results

#results = m_versus_q([10,100,1000,10000,100000,1000000],10)

q = random.uniform(high=0.5,size = 10)
q = sort(q)
set_printoptions(precision=2)
print(q)
print(calc_m(q))


#N = 10000 # Number of variables
#m = 30 # Number of unbalanced variables
#simulate_estimation_of_q_and_m(N,m,.05,10000)

# how does m scale with N (assuming q_i drawn randomly from 0,.5
    









