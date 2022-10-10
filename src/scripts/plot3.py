import json
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130

ini = 1951
fin = 2100
step = 1
prec = 1000

xs = list(range(ini,fin+1,step))

fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,5))

data = dict()
with open(f'cache/stocks/lang/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['s'] = json.load(f)
with open(f'cache/stocks/lang/recorded_speech_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['rw'] = json.load(f)
with open(f'cache/stocks/lang/internet_users_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['iu'] = json.load(f)
with open(f'cache/stocks/lang/popular_platforms_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['pp'] = json.load(f)
with open(f'cache/stocks/lang/indexed_websites_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['iw'] = json.load(f)
with open(f'cache/stocks/lang/common_crawl_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
    data['cc'] = json.load(f)
#with open(f'cache/stocks/lang/high_quality_{prec}_{prec}_{ini}_{fin}_{step}.pointset') as f:
#    data['hq'] = json.load(f)

colors = {
    's': 'black',
    'iu': '#ff00ff',
    'rw': '#ff0000',
    'cc': '#ff8000',
    'pp': '#0000ff',
    'iw': '#00ffff',
    'hq': 'yellow',
}
labels = {
    's': 'Aggregated model',
    'iu': 'Internet users',
    'rw': 'Recorded speech',
    'cc': 'CommonCrawl',
    'pp': 'Popular platforms',
    'iw': 'Indexed websites',
    'hq': 'High quality',
}

axes = {
    's': ax2,
    'iu': ax1,
    'rw': ax1,
    'cc': ax1,
    'pp': ax1,
    'iw': ax1,
    'hq': ax2,
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
    print(f"{gr[2099-ini][0]:.2%}, {gr[2099-ini][1]:.2%}, {gr[2099-ini][2]:.2%}")
    axes[k].fill_between(xs, p5, p95, alpha=0.1*alpha_correction, color=colors[k])
    axes[k].fill_between(xs, p15, p85, alpha=0.2*alpha_correction, color=colors[k])
    axes[k].fill_between(xs, p25, p75, alpha=0.3*alpha_correction, color=colors[k])
    axes[k].fill_between(xs, p35, p65, alpha=0.4*alpha_correction, color=colors[k])
    axes[k].fill_between(xs, p45, p55, alpha=0.5*alpha_correction, color=colors[k])
    axes[k].plot(xs, p50, color=colors[k], label=labels[k])

ax1.legend()
ax2.legend()

ax1.set_xlabel('Year')
ax1.set_ylabel('Number of words (log)')
ax1.grid(True, which="major", ls="-")
ax1.set_yscale('log')
ax1.set_ylim(1e13,1e19)
ax1.set_xlim(2018,2100)

ax2.set_xlabel('Year')
ax2.set_ylabel('Number of words (log)')
ax2.grid(True, which="major", ls="-")
ax2.set_yscale('log')
ax2.set_ylim(1e13,1e19)
ax2.set_xlim(2018,2100)

plt.savefig('lang_models', dpi=500, format='png')
plt.show()
