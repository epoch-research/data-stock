import json
import pandas as pd
import numpy as np
from math import log10, exp
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130

ini = 1951
fin = 2100
step = 1
prec = 1000

xs = list(range(ini,fin+1,step))

data = dict()
with open(f'cache/stocks/vision/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['ag'] = json.load(f)
with open(f'cache/stocks/vision/popular_platforms_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['pp'] = json.load(f)
with open(f'cache/stocks/vision/external_estimate_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['ex'] = json.load(f)


colors = {
    'ag': 'black',
    'ex': '#ff00ff',
    'pp': '#0000ff',
}
labels = {
    'ag': 'Aggregated model',
    'ex': 'External estimate',
    'pp': 'Popular platforms',
}


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
    gr = [[(a/b)-1 for a,b in zip(qt[t+dt],qt[t])] for t in range(len(f)-dt)]
    return [
        [
            np.quantile(g, 0.05),
            np.quantile(g, 0.5),
            np.quantile(g, 0.95),
        ] for g in gr
    ]

fig, ax = plt.subplots(1,1, figsize=(7,6))

for k,v in data.items():
    if k in {}:
        continue
    alpha_correction = 0.5 if k != 's' else 0.5

    p5 =  quantile(v,0.05)
    p15 = quantile(v,0.15)
    p25 = quantile(v,0.25)
    p35 = quantile(v,0.35)
    p45 = quantile(v,0.45)
    p50 = quantile(v,0.5) 
    p55 = quantile(v,0.55)
    p65 = quantile(v,0.65)
    p75 = quantile(v,0.75)
    p85 = quantile(v,0.85)
    p95 = quantile(v,0.95)
    gr = growth(v, 1)
    print(f"{k}, {p50[2022-ini]:e}, {p5[2022-ini]:e}, {p95[2022-ini]:e}")
    print(f"{gr[2022-ini][0]:.2%}, {gr[2022-ini][1]:.2%}, {gr[2022-ini][2]:.2%}")
    ax.fill_between(xs, p5, p95, alpha=0.1*alpha_correction, color=colors[k])
    ax.fill_between(xs, p15, p85, alpha=0.2*alpha_correction, color=colors[k])
    ax.fill_between(xs, p25, p75, alpha=0.3*alpha_correction, color=colors[k])
    ax.fill_between(xs, p35, p65, alpha=0.4*alpha_correction, color=colors[k])
    ax.fill_between(xs, p45, p55, alpha=0.5*alpha_correction, color=colors[k])
    ax.plot(xs, p50, color=colors[k], label=labels[k])

ax.legend()

ax.set_xlabel('Year')
ax.set_ylabel('Number of images (log)')
ax.grid(True, which="major", ls="-")
ax.set_yscale('log')
ax.set_ylim(7e11,2e14)
ax.set_xlim(2020,2080)
plt.savefig('vision_models', dpi=500, format='png')
plt.show()
