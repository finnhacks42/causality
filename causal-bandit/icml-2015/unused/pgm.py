# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 06:53:48 2016

@author: finn
"""
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
from libpgm.pgmlearner import PGMLearner
import numpy as np
import itertools
import pandas as pd

class DiscretePGM(object):
    def __init__(self):
        self.nodes = {}
    
    def addNode(self,name,values,parents,cprob):
        """     "numoutcomes": <number of possible outcome values>,
            "vals": ["<name of value 1>", ... , "<name of value n>"],
            "parents": ["<name of parent 1>", ... , "<name of parent n>"],
            "children": ["<name of child 1>", ... , "<name of child n>"],   
            "cprob": an m * numoutcomes array, where m is number of joint assingments for the parents.  """
        if parents == None:
            assert sum(cprob) == 1, "probabilities do not sum to 1"
            assert len(cprob) == len(values),"number of columns of cprob do not match numoutcomes"
        else:
            parents = [str(p) for p in parents]
            parent_values = [self.nodes[p]["vals"] for p in parents]
            combinations = list(itertools.product(*parent_values))
            assert len(combinations) == cprob.shape[0], "number of rows in cprob inconsistent with number of parent outcomes"
            assert len(values) == cprob.shape[1],"number of columns of cprob does not match numoutcomes"
            cprob_dict = {}
            for row,c in enumerate(combinations):
                cprob_dict[str(list(c))] = cprob[row]
                assert sum(cprob[row]) == 1, "probabilities do not sum to 1"
            cprob = cprob_dict
        
        self.nodes[name] = {"numoutcomes":len(values),"vals":[str(v) for v in values],"parents":parents,"cprob":cprob,"children":[]}
        
    def construct(self):
        skel = GraphSkeleton()
        skel.V = self.nodes.keys()
        skel.E = []
        for node, ndata in self.nodes.iteritems():
            if ndata['parents']:
                for p in ndata['parents']:
                    skel.E.append([p,node])
                    self.nodes[p]['children'].append(node)
        for node, ndata in self.nodes.iteritems():
            if len(ndata['children']) == 0:
                ndata['children'] = None
        data = NodeData()
        data.Vdata = self.nodes
        skel.toporder()
        bn = DiscreteBayesianNetwork(skel, data)
        return bn
        
#a method that prints the distribution as a table from building probabilistic graphical models with Python.
def printdist(jd,bn,normalize=False):
    x=[bn.Vdata[i]["vals"] for i in jd.scope]
    zipover=[i/sum(jd.vals) for i in jd.vals] if normalize else jd.vals
    #creates the cartesian product
    k=[a + [b] for a,b in zip([list(i) for i in itertools.product(*x[::-1])],zipover)] #joins results to variable combinations
    k.sort(key = lambda x:str(x[:-1]))
    df=pd.DataFrame.from_records(k,columns=[i for i in reversed(jd.scope)]+['probability'])
    return df

def estimate_distrib(skel,samples,query,evidence):
    learner=PGMLearner()
    bayesnet=learner.discrete_mle_estimateparams(skel,samples)
    tablecpd=TableCPDFactorization(bayesnet)
    fac=tablecpd.condprobve(query,evidence)
    df2=printdist(fac,bayesnet)
    return df2

#learn the marginals from gibbs samples
def gibbs_marginals(tcpd,skel,query,evidence,num_samples=5000,):
    samples=tcpd.gibbssample(evidence,num_samples)
    df2=estimate_distrib(skel,samples,query,evidence)
    return df2

#learn the marginals from random samples
def random_sample_marginals(model,query,evidence,num_samples=5000):
    samples=model.randomsample(num_samples,evidence)
    #print samples
    df2=estimate_distrib(model,samples,query,evidence)
    return df2


def cpd(aGivenb):
    "a_given_b is [P(A=1|B=0),P(A=1|B=1)]" 
    cprob = np.zeros((2,2))
    cprob[:,1] = aGivenb
    cprob[:,0] = 1-cprob[:,1]
    return cprob
    

def confounded_graph(n):
    epsilon = 0.4
    pZ = .5 #P(Z = 1)
    pX0 = .1 #P(X0 = 1) must be <= .5
    pXgivenZ = [.4,.3]   #P(X=1|Z=0),P(X=1|Z=1) 
    pYgivenX0 = [.5-pX0/(1.0-pX0)*epsilon,.5+epsilon,] #P(Y = 1|X0)
    
    pgm = DiscretePGM()
    pgm.addNode('Z',[0,1],None,[1-pZ,pZ])
    pgm.addNode('X0',[0,1],None,[1-pX0,pX0])
    for i in range(1,n):
        pgm.addNode('X'+str(i),[0,1],['Z'],cpd(pXgivenZ))
    pgm.addNode('Y',[0,1],['X0'],cpd(pYgivenX0))
    model = pgm.construct()
    factorization = TableCPDFactorization(model)
    return model,factorization
    
def simple_graph(pz,px1gz,px2gz):
    pgm = DiscretePGM()
    pgm.addNode('Z',[0,1],None,[1-pz,pz])
    pgm.addNode('X1',[0,1],['Z'],cpd(px1gz))
    pgm.addNode('X2',[0,1],['Z'],cpd(px2gz))
    model = pgm.construct()
    factorization = TableCPDFactorization(model)
    return factorization

    
# could probably implement causal queries fairly easily on top of the TableCPDFactorization. 

graph = simple_graph(.2,[.3,.7],[.2,.9])
q1 = graph.condprobve({'X1':''},{'X2':'0'})
print q1.vals
     


#q = {'X1':'0'}
# calculate probability distribution
#result = fn.condprobve(q,{})
#df = printdist(result,model)
#df['gibbs'] = gibbs_marginals(fn,model,q,{})['probability']
#df['rejection']=random_sample_marginals(model,q,{})['probability']


