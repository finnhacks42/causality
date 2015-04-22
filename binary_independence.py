import numpy as np
for p in np.arange(0,1.2,.2):
    for q in np.arange(0,1.2,.2):
        print "----------"
        print (1-p)*(1-q),(1-p)*q
        print p*(1-q), p*q
        print "----------"
