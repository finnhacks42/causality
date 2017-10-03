# This scipt generates a bunch of plots of distributions representing
# the rewards for a hypothetical bandit problem.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

params = [(2,2),(3,7),(1,1),(5,5),(4,2),(3,1)]

# Two subplots, unpack the axes array immediately
x = np.linspace(0,1,100)

f, ax = plt.subplots(1, len(params),sharey=True)
f.patch.set_facecolor('white')
plt.setp(ax, xticks=[0.1, 0.5, 0.9], xticklabels=['a', 'b', 'c'],  yticks=[])
ax[0].set_ylabel("$P(y)$")
ax[0].set_xlabel("$y$")
for indx,(a,b) in enumerate(params):
    mean = a/(a+b)
    ax[indx].plot(x,beta.pdf(x,a,b))
    ax[indx].axvline(x = mean,c="red",linestyle="--")
    ax[indx].set_xticks([mean])
    ax[indx].set_xticklabels(['$\mu_{0}$'.format(indx)])

    
    

#x = [0,1,2]
#y = [90,40,65]
#labels = ['high', 'low', 37337]
#plt.plot(x,y, 'r')
#plt.xticks(x, labels, rotation='vertical')


plt.show()
    

#ax1.plot(x,beta.pdf(x,1,1),'r-', lw=5, alpha=0.6, label='beta pdf')
#ax1.set_title('Sharing Y axis')

#print(beta.pdf(x,1,1))
#plt.show()
