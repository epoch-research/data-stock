import json
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130
plt.rcParams.update({'font.size': 14})

PLOT = 'lq' # lq | hq | v

ini = {'lq': 1951, 'hq': 1951, 'v': 1951}[PLOT]
fin = {'lq': 2100, 'hq': 2100, 'v': 2100}[PLOT]
step = 1
prec = 1000

xs = list(range(ini,fin+1,step))

#fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,5))
fig1 = plt.figure(1, figsize=(7,6))
ax1 = fig1.add_subplot()
fig2 = plt.figure(2, figsize=(7,6))
ax2 = fig2.add_subplot()

data = dict()
if PLOT == 'lq':
    with open(f'projections/stocks/lang/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['ag'] = json.load(f)
    with open(f'projections/stocks/lang/recorded_speech_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['rw'] = json.load(f)
    with open(f'projections/stocks/lang/internet_users_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['iu'] = json.load(f)
    with open(f'projections/stocks/lang/popular_platforms_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['pp'] = json.load(f)
    with open(f'projections/stocks/lang/indexed_websites_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['iw'] = json.load(f)
    with open(f'projections/stocks/lang/common_crawl_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['cc'] = json.load(f)
    with open(f'projections/stocks/lang/high_quality_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['hq'] = json.load(f)
elif PLOT == 'hq':
    with open(f'projections/stocks/lang/high_quality_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['hq'] = json.load(f)
elif PLOT == 'v':
    with open(f'projections/stocks/vision/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['ag'] = json.load(f)
    with open(f'projections/stocks/vision/popular_platforms_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['pp'] = json.load(f)
    with open(f'projections/stocks/vision/external_estimate_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
        data['ex'] = json.load(f)

colors = {
    'ag': 'black',
    'iu': '#ff00ff',
    'rw': '#ff0000',
    'cc': '#ff8000',
    'iw': '#00ffff',
    'hq': 'black',
    'ex': '#ff00ff',
    'pp': '#0000ff',
}
labels = {
    'ag': 'Aggregated model',
    'iu': 'Internet users',
    'rw': 'Recorded speech',
    'cc': 'CommonCrawl',
    'pp': 'Popular platforms',
    'iw': 'Indexed websites',
    'hq': 'High quality',
    'ex': 'External estimate',
}

axes = {
    'ag': ax2,
    'iu': ax1,
    'rw': ax1,
    'cc': ax1,
    'pp': ax1,
    'iw': ax1,
    'ex': ax1,
    #'hq': ax2,
    'hq': ax1,
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

for k,v in data.items():
    if k in {}:
        continue
    alpha_correction = 0.5 if k != 's' else 0.5

    p5 =  quantile(v,0.05)
    #p15 = quantile(v,0.15)
    #p25 = quantile(v,0.25)
    #p35 = quantile(v,0.35)
    #p45 = quantile(v,0.45)
    p50 = quantile(v,0.5) 
    #p55 = quantile(v,0.55)
    #p65 = quantile(v,0.65)
    #p75 = quantile(v,0.75)
    #p85 = quantile(v,0.85)
    p95 = quantile(v,0.95)
    gr = growth(v, 1)
    print(f"{k}, {p50[2022-ini]:e}, {p5[2022-ini]:e}, {p95[2022-ini]:e}")
    print(f"{gr[2022-ini][0]:.2%}, {gr[2022-ini][1]:.2%}, {gr[2022-ini][2]:.2%}")
    print(f"{gr[2099-ini][0]:.2%}, {gr[2099-ini][1]:.2%}, {gr[2099-ini][2]:.2%}")
    axes[k].fill_between(xs, p5, p95, alpha=0.1*alpha_correction, color=colors[k])
    #axes[k].fill_between(xs, p15, p85, alpha=0.2*alpha_correction, color=colors[k])
    #axes[k].fill_between(xs, p25, p75, alpha=0.3*alpha_correction, color=colors[k])
    #axes[k].fill_between(xs, p35, p65, alpha=0.4*alpha_correction, color=colors[k])
    #axes[k].fill_between(xs, p45, p55, alpha=0.5*alpha_correction, color=colors[k])
    axes[k].plot(xs, p50, color=colors[k], label=labels[k])

ymin = {'lq':1e12,'hq':1e12,'v':7e11}[PLOT]
ymax = {'lq':1e19,'hq':1e15,'v':2e14}[PLOT]
xmin = {'lq':2018,'hq':2018,'v':2018}[PLOT]
xmax = {'lq':2080,'hq':2080,'v':2080}[PLOT]

ax1.legend()
ax2.legend()

ax1.set_xlabel('Year')
ax1.set_ylabel(f'Number of {"images" if PLOT == "v" else "words"} (log)')
ax1.grid(True, which="major", ls="-")
ax1.set_yscale('log')
ax1.set_ylim(ymin, ymax)
ax1.set_xlim(xmin, xmax)

ax2.set_xlabel('Year')
ax2.set_ylabel(f'Number of {"images" if PLOT == "v" else "words"} (log)')
ax2.grid(True, which="major", ls="-")
ax2.set_yscale('log')
ax2.set_ylim(ymin, ymax)
ax2.set_xlim(xmin, xmax)

plt.figure(1)
plt.tight_layout()
#plt.savefig(f'{PLOT}_models_1.png', dpi=500, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
plt.figure(2)
plt.tight_layout()
#plt.savefig(f'{PLOT}_models_2.png', dpi=500, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()
