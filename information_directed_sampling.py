import scipy.integrate as integrate
from scipy.stats import beta,bernoulli
from scipy.optimize import minimize_scalar
from numpy import *
from matplotlib.pyplot import *

def kl(p,q):
    return p*log(p/q) + (1-p)*log((1-p)/(1-q))

def M_ij(i,j):
    # calculate inner integral for each value of t
    f_inner = x*pdfs[j]
    F_inner_t = integrate.cumtrapz(f_inner,x,initial=0)
    
    # now do the outer integral and return the result ...
    #f = pdfs[i]*prod(cdfs,axis=0)/cdfs_dezerod[i]*F_inner_t #TODO Check this line - should be dividing by cdfs_dezerod[j] too
    f = pdfs[i]*prod(cdfs,axis=0)*F_inner_t/(cdfs_dezerod[i]*cdfs_dezerod[j])
    return 1.0/a[i]*integrate.simps(f,dx = 1/float(n_samples)) 

class BetaBernoulliIDS:
    def __init__(self, num_arms, prior, model = None, num_samples = pow(2,8)+1):
        """ prior is a 2-value tuple indicating the shape of the beta prior over arm rewards
            you can optionally pass in a model, which should be an array of bernoulli random variables of length num_arms"""
        if model:
            assert len(model) == num_arms
            self.model = model
        else:
            self.model = [bernoulli(p) for  p in beta.rvs(prior[0],prior[1],size=num_arms)] # the true reward probability for each arm
        self.K = num_arms
        self.n = num_samples
        self.best_reward = max([p.mean() for p in self.model])
        self.beta1 = zeros(self.K)+prior[0] # number of successes + 1 
        self.beta2 = zeros(self.K)+prior[1] # number of fails + 1
        self.x = linspace(0,1,num_samples)
        self.posteriors = [beta(self.beta1[i],self.beta2[i]) for i in range(self.K)]
        self.pdfs = asarray([p.pdf(self.x) for p in self.posteriors]) # creats a num_arms * n_samples array, where each row is the pdf for an arm
        self.cdfs = asarray([p.cdf(self.x) for p in self.posteriors])
        self.cdfs_dezerod = copy(self.cdfs)
        self.cdfs_dezerod[self.cdfs==0] = 1
        self.F = prod(self.cdfs,axis=0) # n_samples array
        self.a = [integrate.simps(self.pdfs[i]*self.F/self.cdfs_dezerod[i],dx = 1/float(self.n)) for i in range(self.K)]
        self.Q = asarray([integrate.cumtrapz(self.x*self.pdfs[i],self.x,initial=0) for i in range(self.K)]) # num_arms*n_samples array
        self.M = zeros((num_arms,num_arms))
        self.calculate_M()

    
    def calculate_M(self):
        """ calculate M """
        for i in range(self.K):
            f1 = self.pdfs[i]*self.F/self.cdfs_dezerod[i]
            ai = 1.0/float(self.a[i])
            dx = 1.0/float(self.n)
            for j in range(self.K):
                if i == j:
                    self.M[i,j] = ai*integrate.simps(f1*self.x,dx=dx)
                else:
                    self.M[i,j] = ai*integrate.simps(f1*self.Q[j]/self.cdfs_dezerod[j],dx=dx)
    
    def dezero(self,row):
        for i in range(self.n):
            v = self.cdfs_dezerod[row,i]
            if v == 0:
                self.cdfs_dezerod[row,i] = 1
            else:
                break
    
    def update(self,action,y):
        """ update everything that changes due to playing action i and observing reward y """
        if y == 1:
            self.beta1[action]+=1
        else:
            self.beta2[action]+=1

        self.posteriors[action] = beta(self.beta1[action],self.beta2[action])    
        self.pdfs[action] = self.posteriors[action].pdf(self.x)
        self.cdfs[action] = self.posteriors[action].cdf(self.x)
        self.cdfs_dezerod[action] = self.posteriors[action].cdf(self.x)
        self.dezero(action)
        self.F = prod(self.cdfs,axis=0) # n_samples array
        self.Q[action] = integrate.cumtrapz(self.x*self.pdfs[action],self.x,initial=0)
        self.a = [integrate.simps(self.pdfs[k]*self.F/self.cdfs_dezerod[k],dx = 1/float(self.n)) for k in range(self.K)]
        self.calculate_M()
        #print "beta1",self.beta1
        #print "beta2",self.beta2
        #print "pdfs",self.pdfs
        #print "cdfs",self.cdfs
        #print "dezerod",self.cdfs_dezerod
        #print "F",self.F
        #print "a",self.a
        #print "Q",self.Q
        
    def calculate_information_gain(self):
        """ returns the information gain for each arm """
        g = [kl(self.M[:,i],(self.beta1[i])/float(self.beta1[i]+self.beta2[i])).dot(self.a) for i in range(self.K)]
        return g
        
        
    def calculate_expected_regret(self):
        """ returns the expected regret for each arm """
        est_p_best = sum([self.a[i]*self.M[i,i] for i in range(self.K)])
        delta = [est_p_best - (self.beta1[i])/float(self.beta1[i]+self.beta2[i]) for i in range(self.K)]
        return delta
        
        
    def find_best_policy(self,g,d):
        """ returns an optimal policy for choosing the next action given the estimated gain and regret """
        """ returns a pair of arms and the probability of playing the first one.
        equivelent to returning  a vector of probabilities for all the arms as we know that at most 2 of them are non-zero"""
        # we know the best policy has at most two non-zero components (see paper)
        best_ijq = (0,0,-1)
        min_info = None
        for i in xrange(self.K):
            for j in xrange(self.K):
                if i != j:
                # minimize (q*d[i]+(1-q)*d[j])^2/(q*g[i]+(1-q)*g[j]) subject to q in [0,1]
                    objective = lambda q: pow(q*d[i]+(1-q)*d[j],2)/(q*g[i]+(1-q)*g[j])
                    res = minimize_scalar(objective,bounds=(0,1),method='bounded')
                    if min_info is None or res.fun < min_info:
                        min_info = res.fun
                        best_ijq = (j,i,res.x)
        assert best_ijq[2] >=0, "policy probability smaller than 0:"+str(best_ijq[2])           
        return best_ijq
    
    def sample_action(self,policy):
        """ returns an action sampled from the policy """
        action_indx = bernoulli.rvs(policy[2])
        return policy[action_indx]
    
    def do(self,action):
        """ runs the specified action and returns the reward """
        return self.model[action].rvs()      
        
    def plot_posteriors(self):
        """ plots the posteriors over the rewards for each arm (max first 10) """
        fig,ax = subplots(1,min(self.K,10),figsize=(15,5))
        for indx in range(len(ax)):
            ax[indx].plot(self.x,self.pdfs[indx],label=indx)
            ax[indx].axvline(self.model[indx].mean(),color="red")
            ax[indx].set_title("Posterior over $p_"+str(indx+1)+"$")
            ax[indx].set_xlabel("$p_"+str(indx+1)+"$")
            ax[indx].set_ylabel("$f_"+str(indx+1)+"(p_"+str(indx+1)+")$")
        show()
        
    def run(self,horizon):
        regret = zeros(horizon)
        for t in range(horizon):
            gain = self.calculate_information_gain()
            delta = self.calculate_expected_regret()
            policy = self.find_best_policy(gain,delta)
            action = self.sample_action(policy)
            regret[t]= self.best_reward - self.model[action].mean() # the expected reward from choosing this action given the model
            reward = self.do(action)
            self.update(action,reward)
        regret = cumsum(regret)
        return regret
        
        
    def verbose_run(self,horizon,do_plot=True):
        regret_t = zeros(horizon) # records the expected regret the algorithm suffers as a function of time
        arm_value_t = zeros((horizon,self.K)) # records the mean of the posterior distribution for each arm
        gain_t = zeros((horizon,self.K))
        gain_ratio_t = zeros((horizon,self.K))
        played_t = zeros((horizon,self.K))
        
        for t in range(horizon):
            gain = self.calculate_information_gain()
            delta = self.calculate_expected_regret()
            policy = self.find_best_policy(gain,delta)
            action = self.sample_action(policy)
            regret_t[t] = self.best_reward - self.model[action].mean() # the expected reward from choosing this action given the model
            played_t[t,action] = 1
            for k in range(self.K):
                arm_value_t[t,k] = self.posteriors[k].mean()
                gain_t[t,k] = gain[k]
                gain_ratio_t[t,k] = pow(delta[k],2)/float(gain[k])
            reward = self.do(action)
            self.update(action,reward)
        played_t = cumsum(played_t,axis=0)
            
        if do_plot:
            t = range(horizon)
            fig,ax = subplots(2,2,figsize=(15,10)) 
            ax[0,0].plot(t,arm_value_t)
            ax[0,0].set_ylabel("expected return")
            ax[1,0].plot(t,gain_t)
            ax[1,0].set_ylabel("information gain")
            ax[0,1].plot(t,gain_ratio_t)
            ax[0,1].set_ylabel("informatin ratio")
            ax[1,1].plot(t,played_t)
            ax[1,1].set_ylabel("times played")
            xlabel("t")
            show()
            
            plot(t,cumsum(regret_t))
            ylabel("regret")
            xlabel("t")
            show()
        return (regret_t,arm_value_t)

class ThompsonBandit:
    def __init__(self,num_arms,prior,model=None):
        if model:
            assert len(model) == num_arms
            self.model = model
        else:
            self.model = [bernoulli(p) for  p in beta.rvs(prior[0],prior[1],size=num_arms)]
        self.posteriors = [beta(prior[0],prior[1]) for k in range(num_arms)]
        self.best = max([p.mean() for p in self.model])
        
    
    def run(self,horizon):
        regret = zeros(horizon)
        for t in range(horizon):
            samples = [x.rvs() for x in self.posteriors]
            action = argmax(samples)
            regret[t] = self.best - self.model[action].mean()
            reward = self.model[action].rvs()
            params = self.posteriors[action].args
            self.posteriors[action] = beta(params[0]+reward,params[1]+(1-reward))
        regret = cumsum(regret)
        return regret

num_arms = 5
prior = (1,1)
trials = 100
horizon = 100

ids_results = zeros((trials,horizon))
thompson_results =zeros((trials,horizon))

for trial in range(trials):
    model = [bernoulli(p) for p in beta.rvs(prior[0],prior[1],size=num_arms)]
    ids = BetaBernoulliIDS(num_arms,prior,model)
    thompson = ThompsonBandit(num_arms,prior,model)
    ids_results[trial,:] = ids.run(horizon)
    thompson_results[trial,:] = thompson.run(horizon)
    print "Completed trail:",trial
    
t = range(horizon)
f,ax = subplots(1,1,figsize=(15,10))
ax.plot(t,mean(ids_results,axis=0),label="IDS")
ax.plot(t,mean(thompson_results,axis=0),label="Thompson")
ax.legend(loc="lower right")
ax.set_xlabel("Time period")
ax.set_ylabel("Cumulative Expected Regret")
show()
