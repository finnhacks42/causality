from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from math import log
from scipy.stats import entropy

x = np.zeros(10)
y = np.zeros(10)
y[1] = 3
y[2] = -3
z = np.maximum(x,y)
print x
print y
print z
