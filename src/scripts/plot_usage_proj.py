import json
from math import log10, floor, ceil, log, exp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130
plt.rcParams.update({'font.size': 13})

STOCK = 's' # hq | s | v

ini = {'s': 1951, 'hq': 2022, 'v': 1951}[STOCK]
fin = {'s': 2100, 'hq': 2030, 'v': 2100}[STOCK]
step = {'s': 1, 'hq': 0.1, 'v': 1}[STOCK]
prec = 1000

def comp_transform_lang(c):
    return [10**((c - log10(6*20))/2 + log10(20))]

def comp_transform_vis(c):
    return [10**((c - log10(6))/2 - (x/5)) for x in range(10,16)]

comp_transform = {'s': comp_transform_lang, 'hq': comp_transform_lang, 'v': comp_transform_vis}[STOCK]

data = dict()
if STOCK == 'v':
    with open(f'projections/datasets/vision_datasets_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['hist'] = json.load(f)
else:
    with open(f'projections/datasets/language_datasets_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['hist'] = json.load(f)

ini = max(ini, 2022)
fin = min(fin, 2080)
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

data = {k:cumsum(v) for k,v in data.items()}

with open(f'data/compute_projection.csv') as f:
    dt = pd.read_csv(f, header=None, names=['n','c','y'])
    dt.c = dt.c.map(comp_transform)
    dt = dt.explode('c').astype(float)
    groups = dt[['c','y']].groupby('y')
    data['comp'] = groups

def quantile(f, q):
    return [interpolate(t[1], t[0], q) for t in f]

data2 = {}

for k,v in data.items():
    print(k)
    if k in {"comp"}:
        qfn = lambda v,q: v.quantile(q)['c']
        mask = lambda x: x
    else:
        mask = {'s': lambda x: x[71:(71+59)],
                'hq': lambda x:x,
                'v': lambda x: x[71:(71+59)]}[STOCK]
        qfn = lambda v,q: [10**x for x in quantile(v,q)]
    data2[k] = [mask(list(qfn(v,q))) for q in np.linspace(0,1,101)]

print("Got quantiles")

data2["comp"] = [[
    interpolate(xs2, q, x)
    for x in xs] for q in data2["comp"]]

data3 = {}

for stock in [STOCK]:
    for demand in ['comp','hist']:
        print(stock,demand)
        quantiles = []
        #for qstock in data2[stock]:
        for qdemand in data2[demand]:
            quantiles.append(qdemand)

        data3[stock+'_'+demand] = list(zip(*[np.quantile(x,np.linspace(0.05,0.95,10)) for x in zip(*quantiles)]))
    
colors = {
    's_hist': '#ff0000',
    's_comp': '#0000ff',
    'hq_hist': '#ff0000',
    'hq_comp': '#0000ff',
    'v_hist': '#ff0000',
    'v_comp': '#0000ff',
}
labels = {
    's_hist': 'Extrapolation from trend',
    's_comp': 'Extrapolation based on compute',
    'hq_hist': 'Extrapolation from trend',
    'hq_comp': 'Extrapolation based on compute',
    'v_hist': 'Extrapolation from trend',
    'v_comp': 'Extrapolation based on compute',
}

fig, ax = plt.subplots(1,1, figsize=(7,6))

for k,v in data3.items():
    print(k)
    alpha_correction=0.5
    ax.fill_between(xs, v[0], v[-1], alpha=0.1*alpha_correction, color=colors[k])
    ax.plot(xs, [(a+b)/2 for a,b in zip(v[4],v[5])], color=colors[k], label=labels[k])

min_y = {'s': 1e12, 'hq': 3e12, 'v':1e9}[STOCK]
max_y = {'s': 1e19, 'hq': 3e13, 'v':1e15}[STOCK]
min_x = {'s': 2022, 'hq': 2022, 'v':2022}[STOCK]
max_x = {'s': 2080, 'hq': 2026, 'v':2060}[STOCK]

ax.legend(loc="upper left")

ax.set_xlabel('Year')
ax.set_ylabel(f'Number of {"images" if STOCK == "v" else "words"} (log)')
ax.grid(True, which="major", ls="-")
ax.set_yscale('log')
#ax.set_ylim(min_y, max_y)
ax.set_xlim(min_x, max_x)
if(STOCK == 'hq'):
    ax.set_xticks(list(range(2022,2027)))

plt.tight_layout()
#plt.savefig({'s': 'lang_proj_lq.png', 'hq': 'lang_proj_hq.png', 'v':'vision_proj.png'}[STOCK],
#    dpi=500, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

