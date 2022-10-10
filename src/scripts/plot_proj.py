import json
from math import log10
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 180

ini = 1951
fin = 2100
step = 1
prec = 1000

xs = list(range(ini,fin+1,step))
xs2 = np.linspace(2022,2080,100)

def comp_transform_lang(c):
    return 10**((c - log10(6*20))/2 + log10(20))


data = dict()
with open(f'cache/stocks/lang/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['s'] = json.load(f)
with open(f'cache/stocks/lang/high_quality_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['hq'] = json.load(f)
with open(f'cache/datasets/language_datasets_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['hist'] = json.load(f)


def cumsum(f):
    def cs(t):
        xs = list(map(float, t["xs"]))
        ys = list(map(float, t["ys"]))
        cys = [ys[0]]
        for i,y in enumerate(ys[1:]):
            cys.append(cys[-1]+y*(xs[i+1]-xs[i]))
        return [10**x for x in xs], cys[1:]
    return [cs(t) for t in f]

data = {k:cumsum(v) for k,v in data.items()}

with open(f'data/compute_projection.csv') as f:
    dt = pd.read_csv(f, header=None, names=['n','c','y'])
    dt.c = dt.c.map(comp_transform_lang)
    groups = dt[['c','y']].groupby('y')
    data['comp'] = groups

def quantile(f, q):
    return [t[0][np.searchsorted(t[1],q)] for t in f]

colors = {
    's': 'black',
    'hq': 'green',
    'hist': '#ff0000',
    'comp': '#0000ff',
}
#    's': 'black',
#    'iu': '#ff00ff',
#    'rw': '#ff0000',
#    'cc': '#ff8000',
#    'pp': '#0000ff',
#    'iw': '#00ffff',
labels = {
    's': 'Low-quality data stock',
    'hq': 'High-quality data stock',
    'hist': 'Training dataset - historical projection',
    'comp': 'Training dataset - compute projection',
}

fig, ax = plt.subplots(1,1, figsize=(7,6))

for k,v in data.items():
    print(k)
    if k in {"comp"}:
        x = xs2
        qfn = lambda v,q: v.quantile(q)['c']
    else:
        x = xs
        qfn = quantile
    alpha_correction = 1#0 if k != 's' else 0.5

    p5 =  qfn(v,0.05)
    p15 = qfn(v,0.15)
    p25 = qfn(v,0.25)
    p35 = qfn(v,0.35)
    p45 = qfn(v,0.45)
    p50 = qfn(v,0.5) 
    p55 = qfn(v,0.55)
    p65 = qfn(v,0.65)
    p75 = qfn(v,0.75)
    p85 = qfn(v,0.85)
    p95 = qfn(v,0.95)
    ax.fill_between(x, p5, p95, alpha=0.1*alpha_correction, color=colors[k])
    #ax.fill_between(x, p15, p85, alpha=0.2*alpha_correction, color=colors[k])
    #ax.fill_between(x, p25, p75, alpha=0.3*alpha_correction, color=colors[k])
    #ax.fill_between(x, p35, p65, alpha=0.4*alpha_correction, color=colors[k])
    #ax.fill_between(x, p45, p55, alpha=0.5*alpha_correction, color=colors[k])
    ax.plot(x, p50, color=colors[k], label=labels[k])

ax.legend()

ax.set_xlabel('Year')
ax.set_ylabel('Number of words (log)')
ax.grid(True, which="major", ls="-")
ax.set_yscale('log')
ax.set_ylim(1e12,1e20)
ax.set_xlim(2022,2040)
plt.savefig('lang_projection_svg', dpi=500, format='svg')
#plt.show()
