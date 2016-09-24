# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:24:25 2016

@author: finn
"""

        #self.epsilon = epsilon
        #self.epsilon2 = self.pX[1,0]/self.pX[0,0]*self.epsilon

class VeryConfounded(Model):
    def __init__(self,a,b,nZ,q1,q2,pYgivenX):
        self.N = 2 # number of parents of Y
        self.nZ = nZ # number of confounding variables
        self.K = 2*(self.N+self.nZ)+1 # z_1 ... z_nz=0,z_1...z_nz=1,x1=0,x2=0,x1=1,x2=1,do()
        self.q1,self.q2 = q1,q2
        self.a,self.b = a,b
        self.pygx = pYgivenX.reshape((2,2)) # 2*2 matrix [[p(y|0,0),p(y|0,1)],
                                                           #  [p(y|1,0),p(y|1,1)]] 

        self.pX2 = np.full(self.K,(1-q1)*b) # vector of length K, P(X2=1|a) for each action a
        self.pX2[nZ:] = (1-q1)*b+q1*q2**(nZ-2)
        self.pX2[0] = b
        self.pX2[nZ] = q2**(nZ-1)
        self.pX2[[-5,-3,-1]] = (1-q1)*b +q1*q2**(nZ-1)# for do(x1=0),do(x1=1),do() 
        self.pX2[-2] = 1 # for do(x2 = 1)
        self.pX2[-4] = 0 # for do(x2 = 0)
        
        
        
        self.pX2 = np.vstack((1-self.pX2,self.pX2))
        
        self.pre_compute()
        
    def sample_estimate_P(self,samples):
        
        return
            
 
    
    def P(self,x):
        px2 = self.pX2[x[1]] # probability X2 = x2
                
        
        return self.pX1givenA[x[0]]*self.pX2[x[1]] 
        
    def pYgivenX(self,x):
        return self.pygx[x[0],x[1]]
        
    def sample(self,action):
        z = binomial(1,self.q2,size=self.nZ)
        z[0] = binomial(1,self.q1)
        
        if action < 2*self.nZ: # setting one of the z's
            i,j = action % self.nZ, action/self.nZ
            z[i] = j
            

        s = z[1:].mean()
        p = z[1:].prod()
        
        
        if z[0] == 0:
            x1 = binomial(1,self.a)
            x2 = binomial(1,self.b)
        else:
            x1 = binomial(1,s)
            x2 = binomial(1,p)
            
        if not action < 2*self.nZ: # setting one of the x's (or nothing)
            if action == self.K - 2:
                x2=1
            elif action == self.K - 3:
                x1=1
            elif action == self.K - 4:
                x2=0
            elif action == self.K - 5:
                x1=0
                
            
        x = np.asarray([x1,x2])
        y = binomial(1,self.pYgivenX(x))
        return x,y

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
    
class TestModelVeryConfoundedEquivalence(unittest.TestCase):
    def setUp(self):
        Nz = 3
        pZ1 = .1
        pZ = .4
        a = .3
        b = .7
        py = np.asarray([.1,.5,.3,.2])
    
        self.model1 = GeneralModel.create_very_confounded(Nz,pZ1,pZ,a,b,py)
        self.model2 = VeryConfounded(a,b,Nz,pZ1,pZ,py)
        
        px1givenA = []
        px2givenA = []
        for m in self.model1.post_action_models:
            infer = VariableElimination(m)
            marginals,_ = infer.query(['X1','X2'])
            px1givenA.append(marginals['X1'].reduce([('X1',1)],inplace=False).values)
            px2givenA.append(marginals['X2'].reduce([('X2',1)],inplace=False).values)        
        
        self.px1givenA = np.asarray(px1givenA)
        self.px2givenA = np.asarray(px2givenA)
        
             
    def test_px1givenA(self):
        np_test.assert_array_almost_equal(self.px1givenA,self.model2.pX1givenA[1])
        
    def test_px2givenA(self):
        np_test.assert_array_almost_equal(self.px2givenA,self.model2.pX2givenA[1])
        
    def test_P_simulate(self): #TODO
        
    def test_P(self):
        #for x in self.model1.generate_binary_assignments():
        x = np.asarray([0,0])
        np_test.assert_array_almost_equal(self.model1.P(x),self.model2.P(x),decimal=3)
        
#    def test_rewards(self):
#        np_test.assert_array_almost_equal(self.model1.expected_rewards,self.model2.expected_rewards,decimal=2)
    
