import json
import numpy as np
from matplotlib import pyplot as plt

data = dict()

with open(f'../../projections/stocks/lang/aggregation_1000_1000_1951_2100_1.pointset') as f:
    data['ag'] = json.load(f)
#with open(f'../../projections/stocks/vision/aggregation_1000_1000_1951_2100_1.pointset') as f:
with open(f'../../projections/stocks/vision/external_estimate_1000_1000_1951_2100_1.pointset') as f:
    data['v'] = json.load(f)

def integral(xs, ys):
    cs = [0]
    for i in range(len(ys)-1):
        cs.append(cs[-1] + (xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2)
    return [x/cs[-1] for x in cs]

def cumsum(f):
    def cs(t):
        xs = list(map(float, t["xs"]))
        ys = list(map(float, t["ys"]))
        return [10**x for x in xs], integral(xs,ys)
    return [cs(t) for t in f]

data = {k:cumsum(v) for k,v in data.items()}

def quantile(f, q):
    return [t[0][np.searchsorted(t[1],q)] for t in f]

def growth(f, dt):
    qt = list(zip(*[quantile(f,q/100) for q in range(1,100)]))
    gr = [[(a/b) for a,b in zip(qt[t+dt],qt[t])] for t in range(len(f)-dt)]
    #gr = [[(a/b)-1 for a in qt[t+dt] for b in qt[t]] for t in range(len(f)-dt)]
    return [
        [
            np.quantile(g, 0.05),
            np.quantile(g, 0.5),
            np.quantile(g, 0.95),
        ] for g in gr
    ]

grl = growth(data['ag'], 1)
grv = growth(data['v'], 1)

print([np.log10((x)**(1/1)) for x in grl[49+22]])
print([np.log10((x)**(1/1)) for x in grv[49+22]])
print([f"{x**(1/1)-1:.2%}" for x in grv[49+22]])
