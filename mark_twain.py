# 10 essays signed Quintus Curtius Snodgrass. Where they by Twain?
import numpy as np
from scipy.stats import wald #special case of inverse guassian with mu == 1
from scipy.stats import invgauss
from scipy.stats import norm
import matplotlib.pyplot as plt

def z(a):
    return wald.pdf(1-a)


# compare the proportion of three letter words in each authors work
twain = [.225, .262,.217,.240,.230,.229,.235,.217]
snodgrass = [.209,.205,.196,.210,.202,.207,.224,.223,.220,.201]

print np.mean(twain)

# Perform wald test for equality of the means
# Wald tests considers if H0: theta = theta0 can be rejected

# Let theta be the difference in means
theta_hat = np.mean(twain) - np.mean(snodgrass)

# test that theta_hat = 0. Assume theta_hat asymptotically Normal(theta_hat - theta0)/se_hat -> N(0,1)

# estimated standard error
se_hat = np.sqrt(np.var(twain)**2/float(len(twain))+np.var(snodgrass)**2/float(len(snodgrass)))

W = theta_hat/se_hat

# reject H0 when |W| > upper a/2 quantile of the standard N(0,1)
print np.mean(twain),np.mean(snodgrass)
print "Difference in means",theta_hat
print "Estimated standard Error in Difference",se_hat
print "Difference/standard error",W

# How to calculate a 95% confidence interval for the difference in the means ???

# Cumulative Normal and its inverse (which feels like what I care about).
# Don't understand where Wald/InverseGauss fit in ...
plt.subplot(211)
x = np.linspace(0,5,100)
y = 1-norm.cdf(x)
plt.plot(x,y)
plt.subplot(212)
x2 = np.linspace(0.9,0.9999,1000)
y2 = norm.ppf(x2)
plt.plot(x2,y2)
#plt.show()

# p-value would appear to be 2*norm.cdf(W)
p_val = 2*norm.cdf(W) # too small to caclulate


a = np.linspace(.00001,.99,100)
a2 = a/2
zvals = [z(i) for i in a2]
plt.plot(a,zvals)
#plt.show()


# Now do a permutation test. H0 is that our samples of data are drawn from the same distribution

# our test statistic T(X1...Xm,Y1...Yn) = |mean(X) - mean(Y)|, the magnitude of the difference in means

twain.extend(snodgrass)
exceeds_test = 0
trials = 100000
for i in xrange(trials):
    data = np.random.permutation(twain)
    T = np.abs(np.mean(data[0:8])- np.mean(data[8:]))
    if T > theta_hat:
        exceeds_test +=1
p = 1.0/trials*exceeds_test
print "Permutation test based p-value",p
