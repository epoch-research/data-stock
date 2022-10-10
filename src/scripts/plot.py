import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 130

df = pd.read_json('out.json')
#df = df.transpose()

colors = {
    's': 'black',
    'sint': '#ff00ff',
    'srec': '#ff0000',
    'scc': '#ff8000',
    'spp': '#0000ff',
    'siw': '#00ffff',
}
labels = {
    's': 'Aggregate model',
    'sint': 'Internet users',
    'srec': 'Recorded speech',
    'scc': 'CommonCrawl',
    'spp': 'Popular platforms',
    'siw': 'Indexed websites',
}
for c in df.columns:
    if c in {}:
        continue
    alpha_correction = 0.2#0.5 if c != 's' else 1

    data = df[c]
    median = data.explode().groupby(level=0).median()
    p5 = data.apply(lambda x: np.quantile(list(map(float,x)),0.05))
    p15 = data.apply(lambda x: np.quantile(list(map(float,x)),0.15))
    p25 = data.apply(lambda x: np.quantile(list(map(float,x)),0.25))
    p35 = data.apply(lambda x: np.quantile(list(map(float,x)),0.35))
    p45 = data.apply(lambda x: np.quantile(list(map(float,x)),0.45))
    p55 = data.apply(lambda x: np.quantile(list(map(float,x)),0.55))
    p65 = data.apply(lambda x: np.quantile(list(map(float,x)),0.65))
    p75 = data.apply(lambda x: np.quantile(list(map(float,x)),0.75))
    p85 = data.apply(lambda x: np.quantile(list(map(float,x)),0.85))
    p95 = data.apply(lambda x: np.quantile(list(map(float,x)),0.95))
    #p5 = df.quantile(0.05, 1)
    #p95 = df.quantile(0.95, 1)
    #median = df.quantile(0.5, 1)
    plt.fill_between(df.index, p5, p95, alpha=0.1*alpha_correction, color=colors[c])
    plt.fill_between(df.index, p15, p85, alpha=0.2*alpha_correction, color=colors[c])
    plt.fill_between(df.index, p25, p75, alpha=0.3*alpha_correction, color=colors[c])
    plt.fill_between(df.index, p35, p65, alpha=0.4*alpha_correction, color=colors[c])
    plt.fill_between(df.index, p45, p55, alpha=0.5*alpha_correction, color=colors[c])
    plt.plot(df.index, median, color=colors[c], label=labels[c])

plt.legend()

plt.xlabel('Year')
plt.ylabel('Number of words (log)')
plt.grid(True, which="major", ls="-")
plt.yscale('log')
plt.ylim(1e13,1e20)
plt.xlim(2017,2100)
plt.show()
