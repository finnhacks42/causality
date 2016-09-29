# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:47:35 2016

@author: finn
"""
from itertools import product
from functools import reduce
from operator import mul

def prod(lst):
    return reduce(mul,lst)

class Factor(object):
    def __init__(self,variables,card,values):
        self.values = values
        self.variables = variables
        self.card = card
        self._stride = {}
        s = 1
        for i,v in enumerate(variables):
            self._stride[v] = s
            s*=card[i]
            
    def _union_vars(self,other):
        new_vars = self.variables[:]
        new_card = self.card[:]
        overlap = []
        for indx,x in enumerate(other.variables):
            if x not in new_vars:
                new_vars.append(x)
                new_card.append(other.card[indx])
            else:
                overlap.append(x)
        return new_vars,new_card,overlap
        
    def stride(self,v):
        return self._stride.get(v,0)
    
        
    def marginalize(self,variable):
        return
        
    def __mul__(self,other):
        new_vars,new_card,overlap = self._union_vars(other)
        result_length = prod(new_card)
        values = [None]*result_length        
        j = 0
        k = 0        
        assignment = [0 for l in range(len(new_vars))]
        for i in range(0,prod(new_card)):
            try:
                values[i] = self.values[j]*other.values[k]
            except IndexError:
                print i,j,k
                print len(values),len(self.values),len(other.values)
            for l,v in enumerate(new_vars):
                assignment[l] = assignment[l]+1
                if assignment[l] == new_card[l]:
                    j = j - (new_card[l] - 1)*self.stride(v)
                    k = k - (new_card[l] - 1)*other.stride(v)
                else:
                    j = j+self.stride(v)
                    k = k+other.stride(v)
                    break
                
        return Factor(new_vars,new_card,values)
       
        
        
    def __str__(self):
        return str(self.values)
        
    def __repr__(self):
        return self.__str__()

        
class S(object):
    def __init__(self,string):
        self.string = string
    
    def __mul__(self,other):
       return S("({0})*({1})".format(self.string,other.string))
        
    
    def __rmul__(self,other):
        return S("({1})*({0})".format(self.string,other.string))
    
    def __str__(self):
        return self.string
    
    def __repr__(self):
        return self.string.__repr__()

              
pz = Factor(['Z'],[2],[S("1-z"),S("z")])

px1gz = Factor(['Z','X1'],[2,2],[S("1-q10"),S("q10"),S("1-q11"),S("q11")])

px2gz = Factor(['Z','X2'],[2,2],[S("1-q20"),S("q20"),S("1-q21"),S("q21")])

zx1 = pz*px1gz

zx1*px2gz

print zx1


