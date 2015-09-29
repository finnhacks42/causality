import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
    

people = 1000
sample_size = people/2.0
counfounders = 3 # lets assume each confounder is binary.

# if we have enough confounders, at least one is likely to remain unbalanced after randomization

# two models where outcome independent of treatment assignment
# now an additive model. P(Y|C1,C2,C3) = logit(c1+c2+c3)
# an or model P(Y|c1,c2,c3) = .8 if c1 or c2 or c3 else .2


# These models are causing concentration.


# Assume we have an experiment with a binary outcome and run exactly the same on two groups.
# How big I expect the difference to be depends on how many times I've performed the experiment
# and what the prob of success is each time. I expect the difference/number of expermints to fall as trial size increases.
# Assume I see some difference between treatment and control
# Ok seems I need to compare proportions and use a chi-squared test.

# create a matrix with one row per person, and a column per counfounder, fill it with random binary values.

# depends how many counfounders there are, how they effect the target. 

np.random.seed(1)

data = np.random.binomial(1,.5,(people,counfounders))
m1 = np.random.binomial(1,sigmoid(np.sum(data,axis=1)))
m2 = np.any(data,axis=1)
print sum(m1[0:people/2])/sample_size,sum(m1[people/2:])/sample_size
print sum(m2[0:people/2])/sample_size,sum(m2[people/2:])/sample_size



# null hypothosis is they are the same. p-val is prob of getting a value as extreme as that seen if null is true.
# null hypothosis is that the data are drawn from binomial distributions with the same (unknown) mean
# of 1000 trials, what's the probability 


