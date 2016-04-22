# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:30:16 2016

@author: finn
"""
import sys
sys.path.append("/home/finn/programming/pyscripts/symfer-0.2.1")
import symfer as sym

pA = .6 #Probability that A = 0
pB = np.array([.5,.8]) #Probability that B = 0 given A=0,A=1
pC = np.array([.5,.2])
pY = np.array([[.2,.9],[.6,.8]]) # P(Y|B,C)

A = {'A':['x0','x1']}
B = {'B':['y0','y1']}
C = {'C':['z0','z1']}
Y = {'Y':['y0','y1']}
factors = {}    # an empty dictionary which we'll fill with factors
factors['A'] = sym.Multinom([A],[pA,1-pA])                  # P(A)
factors['B'] = sym.Multinom([B,A],[pB[0],1-pB[0],pB[1],1-pB[1]]) # P(B|A) 
factors['C'] = sym.Multinom([C,A],[pC[0],1-pC[0],pC[1],1-pC[1]]) # P(C|A)
factors['Y'] = sym.Multinom([Y,C,B],[pY[0][0],1-pY[0][0],pY[0][1], 1-pY[0][1],pY[1][0],1-pY[1][0],pY[1][1],1-pY[1][1]]) # P(Y|B,C)