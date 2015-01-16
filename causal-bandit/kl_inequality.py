def kl(x,y):
    return x*log(x/y)+(1-x)*log((1-x)/(1-y))

p = .5
p2 = .4
e = 1

kl(p2+e*p,p2)
kl
