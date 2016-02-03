# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:58:32 2016

@author: finn
"""


    def test_sample(self, simulations):
        xdata = np.zeros((simulations,self.N),dtype=float)
        ydata = np.zeros(simulations)
        for s in xrange(simulations):
            x,y,p = self.sample()
            xdata[s] = x
            ydata[s] = y
        xm = np.mean(xdata,axis=0)
        xse = 3*np.std(xdata,axis=0)/sqrt(simulations)
        xtrue = np.concatenate(([self.pZ],self.pZ*self.pXgivenZ[0] + (1-self.pZ)*self.pXgivenZ[1]))
        print xtrue
        diffs = np.abs(xtrue-xm)
        over = np.sum(diffs > xse)
        p = (1-.997)
        assert over <= (self.N*p + self.N*p*(1-p)),over
        
            def sample_given_action(self,arm):
        k,j = arm%self.N,1-arm/self.N
        assert 0 <= k <= self.N, "k not a variable indx"
        assert j in [0,1], "j must be 0 or 1"
        
        V = np.empty(self.N,dtype=int)
        V[0] = binomial(1,self.pZ)
        V[k] = j
        V[1:] = binomial(1,self.pXgivenZ[V[0]])
        V[k] = j
        Y = binomial(1,self.pYgivenXi[V[self.i]])
        return V,Y
        
