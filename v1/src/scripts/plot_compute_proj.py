import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import log10

def comp_transform_lang(c):
    return [10**((c - log10(6*20))/2 + log10(20))]

comp_transform = comp_transform_lang
comp = None

with open(f'data/compute_projection.csv') as f:
    dt = pd.read_csv(f, header=None, names=['n','c','y'])
    dt.c = dt.c.map(comp_transform)
    dt = dt.explode('c').astype(float)
    groups = dt[['c','y']].groupby('y')
    comp = groups



ini = 2022
fin = 2080
step = 1
prec = 1000
xs = np.linspace(ini, fin, int((fin-ini)/step+1))
xs2 = np.linspace(2022,2080,100)

def interpolate(xs, ys, x):
    if(min(ys) == max(ys)):
        return ys[0]
    if x < xs[0] or x > xs[-1]:
        return 0
    i = np.searchsorted(xs, x)
    if xs[i] == x: return ys[i]
    return ((x-xs[i-1])*ys[i] + (xs[i]-x)*ys[i-1])/(xs[i]-xs[i-1])

def integral(xs, ys, prec = 101):
    ps = list(np.linspace(xs[0], xs[-1], prec))
    cs = [0]
    for i in range(prec-1):
        cs.append(cs[-1] + (ps[i+1]-ps[i])*(interpolate(xs, ys, ps[i]) + interpolate(xs, ys, ps[i+1]))/2)
    return (ps, [x/cs[-1] for x in cs])

def cumsum(f):
    def cs(t):
        xs = list(map(float, t["xs"]))
        ys = list(map(float, t["ys"]))
        return integral(xs, ys)
    return [cs(t) for t in f]

comp = [list((lambda v,q: v.quantile(q)['c'])(comp,q)) for q in np.linspace(0,1,101)]

comp = [[
    interpolate(xs2, q, x)
    for x in xs] for q in comp]

plt.fill_between(xs, comp[5], comp[-5], alpha=0.05, color='black')
plt.plot(xs, comp[50], color='black', label='Compute availability forecast')

STOCK = 'v'
min_y = {'s': 1e12, 'hq': 3e12, 'v':1e9}[STOCK]
max_y = {'s': 1e19, 'hq': 3e13, 'v':1e15}[STOCK]
min_x = {'s': 2022, 'hq': 2022, 'v':2022}[STOCK]
max_x = {'s': 2050, 'hq': 2026, 'v':2060}[STOCK]

plt.legend()

plt.xlabel('Year')
plt.ylabel(f'Compute budget (FLOP)')
plt.grid(True, which="major", ls="-")
plt.yscale('log')
#plt.ylim(min_y, max_y)
#plt.xlim(min_x, max_x)

plt.tight_layout()
plt.show()
