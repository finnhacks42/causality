import cProfile
from multicausesim import compareBandits
o = open("tmp.txt","w")
numArms = 8
px = [0.5]*numArms
pyGivenX = [0.5]*pow(2,numArms)
pyGivenX[0:pow(2,numArms-1)] = [0.6]*pow(2,numArms-1) # think further about this step ...
cProfile.run('compareBandits(1,1,o,px,pyGivenX)')
        
