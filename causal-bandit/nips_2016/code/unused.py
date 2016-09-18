# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:24:25 2016

@author: finn
"""

class ScaleableParallelConfounded(ParallelConfounded):
    def __init__(self,q10,q11,q20,q21,pZ,N1,N2,epsilon):
        ParallelConfounded.__init__(self,q10,q11,q20,q21,pZ,N1,N2,epsilon)
        self.parent_assignments = list(product(range(N1+1),range(N2+1))) # all combinations of vector of length 2 specifying [num x_i: i<= N1 = 1, num x_i: i > N = 1]
        self.pa_given_assignment = np.asarray([self.P_of_reparameterized_assignment(n1,n2) for (n1,n2) in self.parent_assignments])
                
    def P_of_reparameterized_assignment(self,n1,n2):
         pa = np.zeros(7)
         pa[0] = (1-self.pZ)*binom(self.N1-1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2)+self.pZ*binom(self.N1-1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2) # do(X1 = 0)
         pa[1] = (1-self.pZ)*binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2)+self.pZ*binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2) # do(X2 = 0)
         pa[2] = (1-self.pZ)*binom(self.N1-1,self.q10).pmf(n1-1)*binom(self.N2,self.q20).pmf(n2)+self.pZ*binom(self.N1-1,self.q11).pmf(n1-1)*binom(self.N2,self.q21).pmf(n2) # do(X1=1)
         pa[3] = (1-self.pZ)*binom(self.N1,self.q10).pmf(n1)*binom(self.N2-1,self.q20).pmf(n2-1)+self.pZ*binom(self.N1,self.q11).pmf(n1)*binom(self.N2-1,self.q21).pmf(n2-1) # do(X2 = 1)
         pa[4] = binom(self.N1,self.q10).pmf(n1)*binom(self.N2,self.q20).pmf(n2) # do(Z = 0)
         pa[5] = binom(self.N1,self.q11).pmf(n1)*binom(self.N2,self.q21).pmf(n2) # do(Z = 1)
         pa[6] = (1-self.pZ)*pa[4]+self.pZ*pa[5] # do()
         return pa
        
    # group variables for which pXgivenZ0 and pXgivenZ1 are equal
    def V(self,eta):
        """ eta should be a vector of length 7, do(X_i: i < N1) = 0, do(X_i: i >= N1) = 0,  do(X_i: i < N1) = 0, do(X_i: i >= N1) = 0 do(Z=0),do(Z=1),do().
            returns a corresponding vector of length 7"""
        nom = (self.pa_given_assignment**2).T
        denom = np.dot(self.pa_given_assignment,eta)
        r = np.true_divide(nom,denom)
        return r.sum(axis=1)
       
    def unparameterize_eta(self,eta):
        """ transform 7 dimensional eta back to K dimensional eta """
        result = np.hstack((np.full(self.N1,eta[0]/self.N1),
                            np.full(self.N2,eta[1]/self.N2),
                            np.full(self.N1,eta[2]/self.N1),
                            np.full(self.N2,eta[3]/self.N2),
                            eta[4],eta[5],eta[6]))
        #TODO check if this ordering is correct ...
        return result
    
    def parameterize(x,self):
        """ transform x into n1,n2 counts """
        n1 = sum(x[0:self.N1])
        n2 = sum(x[self.N1:])
        return n1,n2 