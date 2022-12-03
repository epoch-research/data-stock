import json
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 150
plt.rcParams.update({'font.size': 14})

domain = ["lang", "lang_hq", "vision"]
growth = ["hist", "comp"]
params = {"lang": {"comp": (2022,2060,10,50), "hist": (2023,2080,5,10)},
         "lang_hq": {"comp": (2022,2030,10,10), "hist": (2022,2030,20,20)},
         "vision": {"comp": (2022,2070,5,20), "hist": (2022,2070,5,20)}}

def rollavg(l, w):
    a = []
    for x in l:
        a.append(l[0])
        l = l[1:]
        if(len(a) > w):
            a = a[1:]
        yield sum(a)/len(a)

shapes = {}
for d in domain:
    for g in growth:
        start, end, precision, smoothing = params[d][g]
        with open(f'results/out_intersect_{d}_{g}_{start}_{end}_{precision}.json', 'r') as f:
            data = json.load(f)
            dt = data['value']['value']['xyShape']
            dt['ys'] = list(rollavg(dt['ys'],smoothing))
            shapes[f"{d}_{g}"] = dt


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

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(8,8))

ax1.set_ylabel("Low-quality\nlanguage stock")

ax1.plot(shapes['lang_hist']['xs'], shapes['lang_hist']['ys'])
ax1.set_xlim(2022, 2060)

ax2.plot(shapes['lang_comp']['xs'], shapes['lang_comp']['ys'])
ax2.set_xlim(2022, 2060)

ax3.set_ylabel("High-quality\nlanguage stock")

ax3.plot(shapes['lang_hq_hist']['xs'], shapes['lang_hq_hist']['ys'])
ax3.set_xlim(2022, 2030)

ax4.plot(shapes['lang_hq_comp']['xs'], shapes['lang_hq_comp']['ys'])
ax4.set_xlim(2022, 2030)

ax5.set_ylabel("Vision stock")

ax5.plot(shapes['vision_hist']['xs'], shapes['vision_hist']['ys'])
ax5.set_xlim(2022, 2070)

ax6.plot(shapes['vision_comp']['xs'], shapes['vision_comp']['ys'])
ax6.set_xlim(2022, 2070)

ax5.set_xlabel("Historical projection")
ax6.set_xlabel("Compute projection")

ax1.set_yticks(ticks=[])
ax2.set_yticks(ticks=[])
ax3.set_yticks(ticks=[])
ax4.set_yticks(ticks=[])
ax5.set_yticks(ticks=[])
ax6.set_yticks(ticks=[])

plt.tight_layout()
plt.savefig('intersections.png', dpi=500, format='png')
plt.show()
