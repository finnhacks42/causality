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

N = 100 # Number of variables
m = 10 # Number of unbalanced variables
q = zeros(N)
q[m-1:] = 1/m

# simulate observering for h rounds and estimating bar(q_i)
delta = .05
h = floor(N*m*log(1/delta))
print(h)
simulations = 1000

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

plt.subplots(1,1)
p = linspace(0,1,50)
v = p*(1-p)
plt.plot(p,v)
plt.show()






