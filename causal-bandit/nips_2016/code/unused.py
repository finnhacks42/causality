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
        
#    def V(self,eta):
#        """ The expected value of R (over x sampled from p(x|a)), for each action """
#        if self.parent_assignments is None:
#            self.parent_assignments = self.generate_binary_assignments()
#        expected_R = np.zeros(self.K)
#        for x in self.parent_assignments:
#            pa = self.P(x)
#            expected_R += pa*self.R(pa,eta)
#        return expected_R
               
#    def _calculate_expected_rewards(self):
#        """ Calculate the expected value of Y (over x sampled from p(x|a)) for each action """
#        expected_reward = np.zeros(self.K)
#        for x in self.parent_assignments:
#            pa = self.P(x)
#            expected_reward += pa*self.pYgivenX(x)
#        return expected_reward
               
def prod_all_but_j2(vector):
    indx = np.range(len(vector))
    joint = np.prod(vector)
    joint_on_v = np.true_divide(joint,vector)
    for j in np.where(np.isnan(joint_on_v))[0]:
        joint_on_v[j] = np.prod(vector[indx != j])
    
    return joint_on_v
    
class VeryConfounded(object):
    def __init__(self,a,b,n,q1,q2,pYgivenW):
        #self.pV = np.vstack((1.0-q,q)) TODO fix this.
        self.n = n # number of V variables
        self.N = self.n + 2 # number of variables total
        self.K = 2*self.N + 1 #v_1...v_n = 0,v_1...v_n=1,w0=0,w1=0,w0=1,w1=1,do()
        self.pYgivenW = pYgivenW # 2*2 matrix p(y|0,0),p(y|0,1),p(y|1,0),p(y|1,1)
        
        self.pW0givenA = np.full(self.K,(1-q1)*a + q1*q2*(n-2)/(n-1.0))
        self.pW0givenA[n:] = (1-q1)*a + q1*(q2*(n-2)/(n-1.0)+1/(n-1.0))
        self.pW0givenA[0] = a
        self.pW0givenA[n] = q2
        self.pW0givenA[[-4,-2,-1]] = (1-q1)*a + q1*q2 # for do(w1=0),do(w1=1),do()
        self.pW0givenA[-3] = 1 # for do(w0 = 1)
        self.pW0givenA[-5] = 0 # for do(w0 = 0)
        
        self.pW1givenA = np.full(self.K,(1-q1)*b)
        self.pW1givenA[n:] = (1-q1)*b+q1*q2**(n-2)
        self.pW1givenA[0] = b
        self.pW1givenA[n] = q2**(n-1)
        self.pW1givenA[[-5,-3,-1]] = (1-q1)*b # for do(w0=0),do(w0=1),do()
        self.pW1givenA[-2] = 1 # for do(w1 = 1)
        self.pW1givenA[-4] = 0 # for do(w1 = 0)
        
        self.pW0givenA = np.vstack((1-self.pW0givenA,self.pW0givenA))
        self.pW1givenA = np.vstack((1-self.pW1givenA,self.pW1givenA))
        self.parent_vals = np.asarray([(0, 0), (0, 1), (1, 0), (1, 1)])
        self.expected_rewards = self.estimate_rewards(100000)
        self.optimal = np.max(self.expected_rewards)
        
    def estimate_rewards(self,samples_per_action):
        total = np.zeros(self.K)
        for s in xrange(samples_per_action):
            for a in range(self.K):
                x,y = self.sample(a)
                total[a] += y
        return total/float(samples_per_action)
    
    def P(self,w):
        return self.pW0givenA[w[0],:]*self.pW1givenA[w[1],:] 
        
    def R(self,x,eta):
        pa = self.P(x)
        Q = (eta*pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
        
    def V(self,eta):
        va = np.zeros(self.K)  
        for x in self.parent_vals:
            pa = self.P(x)
            Q = (eta*pa).sum()
            ratio = np.true_divide(pa**2,Q)
            ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
            va += ratio         
        return va 
        
    def pW0(self,v):
        return v.mean()
    
    def pW1(self,v):
        return v.prod()
    
    def sample(self,action):
        v = binomial(1,q2,size=self.n)
        v[0] = binomial(1,q1)
        if action < 2*self.n: # setting one of the V's
            i,j = action % self.n, action/self.N
            v[i] = j
        w0 = binomial(1,self.pW0(v))
        w1 = binomial(1,self.pW1(v))
        if not action < 2*self.n:     
            if action == self.K - 2:
                w1 = 1
            elif action == self.K - 3:
                w0 = 1
            elif action == self.K - 4:
                w1 = 0
            elif action == self.K - 5:
                w0 = 0
        x = np.zeros(self.N)  
        x[0:self.n] = v
        x[self.n] = w0
        x[self.n+1]= w1
        y = binomial(1,self.pYgivenW[w0,w1])
        return x,y
    
    def sample_multiple(self,actions,n):
        """ sample the specified actions, n times each """
        return binomial(n,self.expected_rewards[actions])
    
    def m(self,eta):
        maxV = self.V(eta).max()
        assert not np.isnan(maxV), "m should not be nan"
        return maxV