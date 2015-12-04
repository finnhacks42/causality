def kl(x,y):
    return x*log(x/y)+(1-x)*log((1-x)/(1-y))

T = 100
epsilon = 0.2
K = 5

# Under enviroment 1, expected reward for arm 0 is 0, for arm 1 is 1/2+e, for all other arms, 1/2

for t in range(T):
    
