from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import norm


sample_size = 100 # how many in each of target and control
confounders = range(0,30)
runs = 1000

data = zeros((runs,len(confounders)))
W = zeros((runs,len(confounders)))
plt.figure()
for row in range(runs):
    for c in confounders:
        target = random.normal(size=(sample_size,c))
        control = random.normal(size=(sample_size,c))
        targetY = prod(target,axis=1)
        controlY = prod(control,axis=1)
        diff = abs(mean(targetY) - mean(controlY)) # difference in means
        se = sqrt(var(targetY)/float(sample_size) + var(controlY)/float(sample_size))
        data[row,c] = diff
        W[row,c]=diff/se
        if row == 0:
            plt.subplot(len(confounders),1,c+1)
            plt.hist(targetY)
            plt.hist(controlY)

plt.show()
         
    

y = mean(data,axis=0)
error = 3*sem(data,axis=0)
plt.figure()
plt.subplot(311)
plt.plot(confounders,y)
plt.fill_between(confounders,y-error,y+error,alpha=.5)


y = mean(W,axis=0)
error = 3*sem(W,axis=0)
plt.subplot(312)
plt.plot(confounders,y,color="green")
plt.fill_between(confounders,y-error,y+error,alpha=.5,color="green")

p = 2*norm.cdf(-W)
y = mean(p,axis=0)
error = 3*sem(p,axis=0)
plt.subplot(313)
plt.plot(confounders,y,color="red")
plt.fill_between(confounders,y-error,y+error,alpha=.5,color="red")
plt.show()
# p-values are all around .5
# so we would expect to get a difference as large as that observed around 1/2 the time


