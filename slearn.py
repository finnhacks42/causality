from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
import numpy as np



def set_uniform_cpds(G, cards = {}):
    """ cards is an optional mapping from variable name to cardinality.
        assumed to be 2 for any variables for which its not specified"""
    for n in G.nodes():
        parents  = G.predecessors(n)
        parents_card = [cards.get(p,2) for p in parents]
        card = cards.get(n,2)
        values = np.full((card,np.prod(parents_card)),1.0/card)
        cpd = TabularCPD(n,card,values,parents,parents_card)
        G.add_cpds(cpd)
                    

    
# now I want to be able to do sampling and updating ...



G = BayesianModel()
G.add_edges_from([("X1","Y"),("X2","Y"),("X3","Y")])
set_uniform_cpds(G)

# idea

#1) sample a model

#2) sample a beleif about the state of that model from current data we have
    # requires posterior not just maximum likelyhood estimates.

    


#3) take best action according to that belief

#4) add result to data set


