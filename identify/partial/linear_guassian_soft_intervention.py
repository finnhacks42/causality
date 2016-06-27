# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:30:22 2016

@author: finn
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm
import seaborn as sns
#np.set_printoptions(precision=1)



    

samples = 100000
mz,mu = 1,3
wx0,wy0 = -10,5
wxz,wxu,wyu,wyx = 2.0,.1,0.5,4.0
vz,vu,vex,vey = 0.1,3.0,0.2,0.5

mx = wx0+wxz*mz+wxu*mu # theoretical mean of P(x)
my = wy0+wyu*mu+wyx*mx # theoretical mean of P(y)

z = np.random.normal(mz,sqrt(vz),size=samples)
u = np.random.normal(mu,sqrt(vu),size=samples)
ex = np.random.normal(0,sqrt(vex),size=samples)
ey = np.random.normal(0,sqrt(vey),size=samples)
x = wx0 + wxz*z+wxu*u+ex
y = wy0+wyu*u+wyx*x+ey



# estimate the covariance matrix for z,x,y
cov = np.cov(np.vstack((z,x,y)))
cor = np.corrcoef(np.vstack((z,x,y)))


czz = vz 
czx = wxz*czz
czy = wyx*czx
cxx = wxz**2*vz+wxu**2*vu+vex
cxy = wxu*wyu*vu+wyx*cxx
cyy = wyu**2*vu+2*wyx*wyu*wxu*vu+wyx**2*cxx+vey

theoretical = [czz,czx,czy,cxx,cxy,cyy]
simulated = [cov[0,0],cov[0,1],cov[0,2],cov[1,1],cov[1,2],cov[2,2]]

for i in range(len(theoretical)):
    print("{:.2f},{:.2f}".format(theoretical[i],simulated[i]))
    
# check the pearson correlation matrix
sz = sqrt(vz)
sx = sqrt(cxx)
sy = sqrt(cyy)
corzx = wxz*sz/sx
corzy = wyx*wxz*sz/sy
corxy = wyx*sx/sy + wxu*wyu*vu/(sx*sy)

theoretical = [corzx,corzy,corxy]
simulated = [cor[0,1],cor[0,2],cor[1,2]]
for i in range(len(theoretical)):
    print("{:.2f},{:.2f}".format(theoretical[i],simulated[i]))
    
    
# approximate p(x|y = -22.5)
rows = (-23 < y) & (y < -22)
xgiveny = x[rows]
D = 1.0/(cxx*cyy - cxy**2)

theory_mean =  mx + D*cxy*cyy*(-22.5-my)
theory_var = cxx - D*cxy*cyy*cxy

# Check variance of P(Y|do(X=x))
z = np.random.normal(mz,sqrt(vz),size=samples)
u = np.random.normal(mu,sqrt(vu),size=samples)
ex = np.random.normal(0,sqrt(vex),size=samples)
ey = np.random.normal(0,sqrt(vey),size=samples)
x = -2
ygdox = wy0+wyu*u+wyx*x+ey

r = czy/czx

vygdox = ygdox.std()**2
vygdox_theory = cyy+r*(r*cxx - 2*cxy)
mygdox_theory = my+r*(x-mx)
print(vygdox,vygdox_theory)
print(ygdox.mean(),mygdox_theory)











#plot f(x) - looks right ...
#f,ax = plt.subplots()
#xt = norm(loc= wx0+wx1*mz+wx2*mu,scale = sqrt((wx1*sz)**2+(wx2*su)**2))
#vals = np.linspace(xt.ppf(0.001),xt.ppf(0.999),100)
#ax.plot(vals,xt.pdf(vals),lw=2)
#ax.hist(x,normed=True,alpha=0.2,bins=30)
#
#f,ax = plt.subplots()
#ax.hist(y,normed=True,bins=30)
#
#f,ax = plt.subplots()
