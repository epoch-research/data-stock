import json
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130

ini = 1951
fin = 2100
step = 1
prec = 1000

xs = list(range(ini,fin+1,step))

data = dict()
with open(f'cache/stocks/lang/aggregation_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['s'] = json.load(f)
with open(f'cache/stocks/lang/recorded_speech_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['rw'] = json.load(f)
with open(f'cache/stocks/lang/internet_users_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['iu'] = json.load(f)
with open(f'cache/stocks/lang/popular_platforms_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['pp'] = json.load(f)
with open(f'cache/stocks/lang/indexed_websites_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['iw'] = json.load(f)
with open(f'cache/stocks/lang/common_crawl_{prec}_{prec}_{ini}_{fin}_{step}.cache') as f:
    data['cc'] = json.load(f)

colors = {
    's': 'black',
    'iu': '#ff00ff',
    'rw': '#ff0000',
    'cc': '#ff8000',
    'pp': '#0000ff',
    'iw': '#00ffff',
}
labels = {
    's': 'Geometric average',
    'iu': 'Internet users',
    'rw': 'Recorded speech',
    'cc': 'CommonCrawl',
    'pp': 'Popular platforms',
    'iw': 'Indexed websites',
}
for k,v in data.items():
    if k in {}:
        continue
    alpha_correction = 0 if k != 's' else 0.5

    p5 = [np.quantile(list(map(float,x)),0.05) for x in v]
    p15 = [np.quantile(list(map(float,x)),0.15) for x in v]
    p25 = [np.quantile(list(map(float,x)),0.25) for x in v]
    p35 = [np.quantile(list(map(float,x)),0.35) for x in v]
    p45 = [np.quantile(list(map(float,x)),0.45) for x in v]
    p50 = [np.quantile(list(map(float,x)),0.5) for x in v]
    p55 = [np.quantile(list(map(float,x)),0.55) for x in v]
    p65 = [np.quantile(list(map(float,x)),0.65) for x in v]
    p75 = [np.quantile(list(map(float,x)),0.75) for x in v]
    p85 = [np.quantile(list(map(float,x)),0.85) for x in v]
    p95 = [np.quantile(list(map(float,x)),0.95) for x in v]
    print(k, p50[2022-ini], p5[2022-ini], p95[2022-ini])
    print(p50[2023-ini]/p50[2022-ini], p5[2023-ini]/p5[2022-ini], p95[2023-ini]/p95[2022-ini])
    plt.fill_between(xs, p5, p95, alpha=0.1*alpha_correction, color=colors[k])
    plt.fill_between(xs, p15, p85, alpha=0.2*alpha_correction, color=colors[k])
    plt.fill_between(xs, p25, p75, alpha=0.3*alpha_correction, color=colors[k])
    plt.fill_between(xs, p35, p65, alpha=0.4*alpha_correction, color=colors[k])
    plt.fill_between(xs, p45, p55, alpha=0.5*alpha_correction, color=colors[k])
    plt.plot(xs, p50, color=colors[k], label=labels[k])

plt.legend()

plt.xlabel('Year')
plt.ylabel('Number of words (log)')
plt.grid(True, which="major", ls="-")
plt.yscale('log')
plt.ylim(1e13,1e20)
plt.xlim(2018,2100)
plt.show()
