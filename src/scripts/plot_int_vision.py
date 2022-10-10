import json
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 180

domain = ["vision"]
growth = ["hist", "comp"]
params = {"vision": {"comp": (2022,2070,5), "hist": (2022,2070,5)}}

shapes = {}
for d in domain:
    for g in growth:
        start, end, precision = params[d][g]
        with open(f'results/out_intersect_{d}_{g}_{start}_{end}_{precision}.json', 'r') as f:
            data = json.load(f)
            shapes[f"{d}_{g}"] =  data['value']['value']['xyShape']

def rollavg(l, w):
    a = []
    for x in l:
        a.append(l[0])
        l = l[1:]
        if(len(a) > w):
            a = a[1:]
        yield sum(a)/len(a)


def integral(xs, ys):
    cs = [0]
    for i in range(len(ys)-2):
        cs.append(cs[-1] + (xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2)
    return [x/cs[-1] for x in cs]

ints = {k:integral(*v.values()) for k,v in shapes.items()}
quantiles = {k:[
    shapes[k]['xs'][np.searchsorted(v, 0.05)],
    shapes[k]['xs'][np.searchsorted(v, 0.5)],
    shapes[k]['xs'][np.searchsorted(v, 0.95)]] for k,v in ints.items()}
print(quantiles)

smoothing = 20

fig, (ax1,ax2) = plt.subplots(2)

ax1.plot(shapes['vision_hist']['xs'], list(rollavg(shapes['vision_hist']['ys'],smoothing)))
ax1.set_xlim(2022, 2070)

ax2.plot(shapes['vision_comp']['xs'], list(rollavg(shapes['vision_comp']['ys'],smoothing)))
ax2.set_xlim(2022, 2070)

ax1.set_xlabel("Historical projection")
ax2.set_xlabel("Compute projection")

ax1.set_yticks(ticks=[])
ax2.set_yticks(ticks=[])

fig.suptitle('Distribution of exhaustion dates')
fig.tight_layout()
plt.show()
