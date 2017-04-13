# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:06:18 2016

@author: finn
"""
import numpy as np
from simulator import Linear
from estimators import StandardEstimator, CausalLLMEstimator, EMEstimator

import matplotlib.pyplot as plt


# create some simulated data

#n = 100
#d = 5
#data_model = Linear(d)
#x_treat,x_control,y_treat,y_control = data_model.sample(n)
#
#
#
#llm = CausalLLMEstimator()
#llm.fit(x_treat,x_control,y_treat,y_control)
#
#logistic = StandardEstimator()
#logistic.fit(x_treat,x_control,y_treat,y_control)
#
#em = EMEstimator()
#em.fit(x_treat,x_control,y_treat,y_control)

from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()














