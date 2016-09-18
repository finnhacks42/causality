# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:39:19 2016

@author: finn
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:51:47 2016

@author: finn
"""
import numpy as np

from math import sqrt,log,ceil

import matplotlib.pyplot as plt
from time import time

import pickle


REGRET_LABEL = "Regret"
HORIZON_LABEL = "T"
M_LABEL = "m(q)"


   
model = PgmpyModel.create_confounded_parallel(3,.1)        
eta,m_min = find_eta(model)
              
def sample_pa(model,p_indx,sims):
    result = []
    for action in range(model.K):
        d = {}
        result.append(d)
        for s in xrange(sims):
            x,y = model.sample(action)
            p = tuple(x[p_indx])
            if p in d:
                d[p] += 1
            else:
                d[p] = 1
        for key,value in d.iteritems():
            d[key] = value/float(sims)
    
    for x in model.parent_vals:
        px = model.P(x)
        spx = [dic.get(tuple(x),0.0) for dic in result]
        print x,px,spx
        
            
    
#regret,mean,error = experiment1(50,1000,.3,TperK = 6,Tstep = 25)
    
def regret_vs_T(N,simulations,epsilon=None,Tstep=None,TperK=10):
    
    a = 9.0
    q = part_balanced_q(N,2) 
    if epsilon:
        model = Parallel(q,epsilon)
        Tmin = 10
        vary_epsilon = False
    else:
        model =  Parallel(q,.5)
        Tmin = int(ceil(4*model.K/a))
        vary_epsilon = True
    
    Tstep = model.K if not Tstep else Tstep
    eta,mq = model.analytic_eta()
     
    T_vals = range(Tmin,TperK*model.K,Tstep)
    
    causal = GeneralCausal()
    causal_parallel = ParallelCausal()
    baseline  = SuccessiveRejects()
    
    ts = time()   
    regret = np.zeros((len(T_vals),4,simulations))
    for s in xrange(simulations):
        if s % 100 == 0:
                print s
        for T_indx,T in enumerate(T_vals): 
            if vary_epsilon: #variable epsilon
                epsilon = sqrt(model.K/(a*T))
                model.set_epsilon(epsilon)
            regret[T_indx,0,s] = causal.run(T,model,eta,mq)
            regret[T_indx,1,s] = causal_parallel.run(T,model)
            regret[T_indx,2,s] = baseline.run(T,model)
            regret[T_indx,3,s] = epsilon
            
    te = time()
    print 'took: %2.4f sec' % (te-ts)
    
    mean = regret.mean(axis=2)       
    error = 3*regret.std(axis=2)/sqrt(simulations)
    
    fig,ax = plt.subplots()
    ax.errorbar(T_vals,mean[:,0],yerr=error[:,0], label="Algorithm 2",linestyle="",marker="s",markersize=4) 
    ax.errorbar(T_vals,mean[:,1],yerr=error[:,1], label="Algorithm 1",linestyle="",marker="o",markersize=5)    
    ax.errorbar(T_vals,mean[:,2],yerr=error[:,2], label="Successive Rejects",linestyle="",marker="D",markersize=4) 
    #ax.plot(T_vals,mean[:,3],label="epsilon")
    ax.set_xlabel(HORIZON_LABEL)
    ax.set_ylabel(REGRET_LABEL)
    ax.legend(loc="upper right",numpoints=1)
    fig_name = "exp_regret_vs_T_N{0}_a{1}_s{2}_{3}.pdf".format(N,a,simulations,now_string())
    fig.savefig(fig_name, bbox_inches='tight') 
    return regret,mean,error  

def worst_case_constant():
    bandit = ParallelCausal()
    T = 400
    N = 50
    sims = 1000
    q = part_balanced_q(N,2)
    model =  Parallel(q,.5)
    a_vals = np.linspace(.1,2,20)
    regret = np.zeros((len(a_vals),sims))
    for indx,a in enumerate(a_vals):
        epsilon = sqrt(2.0/float(a*T))
        model.set_epsilon(epsilon)
        for s in range(sims):
            regret[indx,s] = bandit.run(T,model)
    
    mean = regret.mean(axis=1)
    plt.plot(a_vals,mean)


        

def regret_vs_m(N,epsilon,simulations,T):
    m_vals = range(2,N,2)
    causal = GeneralCausal()
    causal_parallel = ParallelCausal()
    baseline  = SuccessiveRejects()
    
    ts = time()   
    regret = np.zeros((len(m_vals),3,simulations))
    for s in xrange(simulations):
        if s % 100 == 0:
                print s
        for m_indx,m in enumerate(m_vals): 
            q = part_balanced_q(N,m)
            model = Parallel(q,epsilon)
            eta,mq = model.analytic_eta()
            regret[m_indx,0,s] = causal.run(T,model,eta,mq)
            regret[m_indx,1,s] = causal_parallel.run(T,model)
            regret[m_indx,2,s] = baseline.run(T,model)
            
    te = time()
    print 'took: %2.4f sec' % (te-ts)
    
    mean = regret.mean(axis=2)       
    error = 3*regret.std(axis=2)/sqrt(simulations)
    
    fig,ax = plt.subplots()
    ax.errorbar(m_vals,mean[:,0],yerr=error[:,0], label="Algorithm 2",linestyle="",marker="s",markersize=4) 
    ax.errorbar(m_vals,mean[:,1],yerr=error[:,1], label="Algorithm 1",linestyle="",marker="o",markersize=5)    
    ax.errorbar(m_vals,mean[:,2],yerr=error[:,2], label="Successive Rejects",linestyle="",marker="D",markersize=4) 
    ax.set_xlabel(M_LABEL)
    ax.set_ylabel(REGRET_LABEL)
    ax.legend(loc="lower right",numpoints=1)
    fig_name = "exp_regret_vs_m_N{0}_T{1}_s{2}_{3}.pdf".format(N,T,simulations,now_string())
    fig.savefig(fig_name, bbox_inches='tight') 
    return regret,mean,error



def experiment2():                          
    ts = time()
    a,b,n,q1,q2, = 1,1,20,.1,.5
    pYgivenW = np.asarray([[.5,.5],[.5,.6]])   
    model = VeryConfounded(a,b,n,q1,q2,pYgivenW) 
    print model.expected_rewards
    
    eta,m = find_eta(model)
    te = time()
    print 'took: %2.4f sec' % (te-ts)
        
    T_vals = range(model.K,10*model.K,model.K)
    simulations = 100
    causal = GeneralCausal()
    baseline  = SuccessiveRejects()
    
    ts = time()   
    regret = np.zeros((len(T_vals),2,simulations))
    for s in xrange(simulations):
        if s % 100 == 0:
                print s
        for T_indx,T in enumerate(T_vals): 
            regret[T_indx,0,s] = causal.run(T,model,eta,m)
            regret[T_indx,1,s] = baseline.run(T,model)
            
    te = time()
    print 'took: %2.4f sec' % (te-ts)
    
    mean = regret.mean(axis=2)       
    error = 3*regret.std(axis=2)/sqrt(simulations)
    
    fig,ax = plt.subplots()
    ax.errorbar(T_vals,mean[:,0],yerr=error[:,0], label="Algorithm 2",linestyle="",marker="s",markersize=4)    
    ax.errorbar(T_vals,mean[:,1],yerr=error[:,1], label="Successive Rejects",linestyle="",marker="D",markersize=4) 
    ax.set_xlabel(HORIZON_LABEL)
    ax.set_ylabel(REGRET_LABEL)
    ax.legend(loc="upper right",numpoints=1)
    fig_name = "exp_regret_cnfd_vs_T_N{0}_s{1}_{2}.pdf".format(model.N,simulations,now_string())
    fig.savefig(fig_name, bbox_inches='tight') 
    return regret,mean,error



## Experiment 1
#N = 50
#epsilon = .3
#simulations = 100
#T = 400
#regret,mean,error = regret_vs_m(N,epsilon,simulations,T)
#pickle.dump(regret, open("experiment1_{0}.pickle".format(now_string()),"wb"))

# Experiment 2
#N= 50
#simulations = 100
#regret2,mean2,error2 = regret_vs_T(N,simulations,epsilon=None,Tstep=None,TperK=10)
#pickle.dump(regret2, open("experiment2_{0}.pickle".format(now_string()),"wb"))

## Experiment 3
#simulations = 10000
#N = 50
#epsilon = .3
#regret3,mean3,error3 = regret_vs_T(N,simulations,epsilon=epsilon,Tstep=25,TperK=6)
#pickle.dump(regret3, open("experiment3_{0}.pickle".format(now_string()),"wb"))




#q = part_balanced_q(10,2) 
#print q       
#model = Parallel(q,.2)
#eta,m = model.analytic_eta()
#bandit = GeneralCausal()
##T_vals = range(55,70,1)
##sims = 1000
##regret = np.zeros((len(T_vals),sims))
##for tindx,t in enumerate(T_vals):
##    for s in xrange(sims):
##        regret[tindx,s] = bandit.run(t,model,eta,m)
##means = regret.mean(axis=1)
##
##
##plt.plot(T_vals,means)
#
##high at 63, low at 64
#print "eta",eta,m
#bandit.run(63,model,eta,m)
#bandit.run(64,model,eta,m)

#fig_name = "exp_regret_vs_T_N{0}_a{1}_s{2}_{3}.pdf".format(N,a,simulations,now_string())
#fig.savefig(fig_name, bbox_inches='tight') 
    

#results = []
#
#o = open("converge3.txt","w")
#for t in range(30):
#    eta0 = random_eta(model.K)
#    res = minimize(model.m,eta0,bounds = [(0.0,1.0)]*model.K, constraints=({'type':'eq','fun':lambda eta: eta.sum()-1.0}),options={'disp': True},method='SLSQP')#, method='L-BFGS-B',
#    
#    results.append(res)
#    if res.success:
#        result =["%.4f"%float(x) for x in res.x]
#        print result
#        data = "["+",".join(result)+"],"+str(res.x.sum())+", "+str(res.fun)+"\n"
#        o.write(data)
#        o.flush()
#    else:
#        print "FAIL"
#    
#
#results.sort(key=lambda x:x.fun)
#for r in results:
#    print ["%.3f" % float(v) for v in r.jac]


