# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:31:57 2017

@author: finn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

np.random.seed(42)

def sample(n_train, n_test):
    n = n_train+n_test
    x1 = np.random.normal(size=n)
    x2 = x1 + np.random.normal(scale = 0.01,size=n)
    y = x1 + x2 + np.random.normal(scale = 0.5,size=n)
    X = np.stack((x1,x2)).T
    X_train, X_test = X[0:n_train,:],X[n_train:,:]
    y_train, y_test = y[0:n_train],y[n_train:]
    return X_train,X_test,y_train,y_test





n_train = 50
n_test = 50
simulations = 1000
r2 = np.zeros(simulations)
coef = np.zeros((simulations,3))
min_c1 = np.inf
model1 = None
model2 = None
bigest_gap = -np.inf

for indx in range(simulations):
    X_train,X_test,y_train,y_test = sample(n_train,n_test)
    model = LinearRegression()
    y_pred = model.fit(X_train,y_train).predict(X_test)
    error = r2_score(y_pred,y_test)
    r2[indx] = error
    coef[indx,0] = error
    coef[indx,1:] = model.coef_
    if model.coef_[0] < min_c1:
        min_c1 = model.coef_[0]
        model1 = model
    if model.coef_[0] - min_c1 > bigest_gap:
        bigest_gap = model.coef_[0] - min_c1
        model2 = model
        
_,X,_,y = sample(1,50)
y1 = model1.predict(X)
y2 = model2.predict(X)
print model1.coef_
print model2.coef_

data = pd.DataFrame({"r2":r2,"b_1":coef[:,1],"b_2":coef[:,2]})
d = pd.melt(data)
sns.boxplot(x = "variable",y="value",data = d)



fig, ax = plt.subplots(1,3)
ax[0].set_title("distribution of $\\beta_1$")
ax[1].set_title("distribution of $\\beta_2$")
ax[2].set_title("distribution of $R^2$")
data.b_1.hist(ax=ax[0],normed=True,range=(-5,5))
data.b_2.hist(ax=ax[1],normed=True,range=(-5,5))
data.r2.hist(ax=ax[2],normed=True,range=(0,1))
fig.savefig("/home/finn/phd/causality/talks/colinear0.png")



fig, ax = plt.subplots(1,1)
ax.plot(y1,label="model1, y = {0:.1f}$x_1$ {1:=+.1f}$x_2$".format(*model1.coef_))
ax.plot(y2,label="model2, y = {0:.1f}$x_1$ {1:=+.1f}$x_2$".format(*model2.coef_))
ax.plot(y,label="true")
ax.legend(loc="upper left")
fig.savefig("/home/finn/phd/causality/talks/colinear1.png")

joint = pd.DataFrame({"$x_1$":X[:,0],"$x_2$":X[:,1]})
#sns.jointplot("$x_1$","$x_2$",data=joint)

fig,ax = plt.subplots(1,1)
sns.regplot("$x_1$","$x_2$",data=joint, fit_reg=False)
    
#g = sns.jointplot("sepal_width", "petal_length", data=iris,
#...                   kind="kde", space=0, color="g")


# cross validate - depending on the data used, the coefficients may vary a lot but the model will fit well regardless. 




