from utils import *
from scipy.optimize import curve_fit

plt.rcParams['figure.dpi'] = 150

fig, [ax1, ax2] = plt.subplots(2,1, figsize=(5,7))

def expmodel(x, A, x0, k, b):
    return A*np.exp(k*(x-x0)) + b

def sigmoid(x, l, x0, k, b):
    return (l / (1 + np.exp(-k*(x-x0))) + b)

def sigmoid2(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))

def sigmexp(x, A, x0e, ke, be, x0s, ks,):
    return expmodel(x, A, x0e, ke, be) * sigmoid2(x, x0s, ks)

def linear(x, m, n):
    return x*m + n

dt = pd.read_csv('data/reddit_open_web_text.csv')
#dt = pd.read_csv('hackernews_submissions.csv')
dt['date'] = dt.year + dt.month/12


p0s = [np.max(dt['size']), np.median(dt.date), 1, np.min(dt['size'])]
boundss = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
p0s2 = [np.median(dt.date), 1]
p0e = [np.median(dt['size']), np.median(dt.date), 1, np.min(dt['size'])]
boundse = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
boundsse = ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
p0l = [1,0]

popt, pcov = curve_fit(lambda a,b,c,d,e: np.log(sigmoid(a,b,c,d,e)), dt.date, np.log(dt['size']), p0s, method='trf', bounds=boundss)

poptexp, pcovexp = curve_fit(lambda a,b,c,d,e: np.log(expmodel(a,b,c,d,e)), dt.date, np.log(dt['size']), p0e, method='trf', bounds=boundse)

poptsexp, pcovsexp = curve_fit(lambda *x: np.log(sigmexp(*x)), dt.date, np.log(dt['size']), p0e+p0s2, method='trf', bounds=boundsse)

poptl, pcovl = curve_fit(lambda a,b,c: np.log(linear(a,b,c)), dt.date, np.log(dt['size']), p0l, method='lm')

#poptsexp = [15107097.792965181, 2010.010843676123, 0.1419728385545392, -9618159.54878189, 2010.5094351626358, 0.9981516671040477]
#poptsexp = list(poptexp)+list(popt[1:3])

#print("Sigm:",popt)
#print("Exp:",poptexp)
#print("Sigm*Exp:",poptsexp)

ax1.plot(dt.date, dt['size'], label='Real data', color=extended_colors[0])
ax1.plot(dt.date, [expmodel(x, *poptexp) for x in dt.date], label='Exponential', color=extended_colors[3])
ax1.plot(dt.date, [sigmoid(x, *popt) for x in dt.date], label='Sigmoid', color=extended_colors[7])
ax1.plot(dt.date, [sigmexp(x, *poptsexp) for x in dt.date], label='Exponential times sigmoid', color=extended_colors[11])
#ax1.plot(dt.date, [linear(x, *poptl) for x in dt.date], label='linear fit')

ax2.plot(dt.date, dt['size'], label='Real data', color=extended_colors[0])
ax2.plot(dt.date, [expmodel(x, *poptexp) for x in dt.date], label='Exponential', color=extended_colors[3])
ax2.plot(dt.date, [sigmoid(x, *popt) for x in dt.date], label='Sigmoid', color=extended_colors[7])
ax2.plot(dt.date, [sigmexp(x, *poptsexp) for x in dt.date], label='Exponential times sigmoid', color=extended_colors[11])
#ax2.plot(dt.date, [linear(x, *poptl) for x in dt.date], label='linear fit')

ax2.set_xlabel('Year')
ax2.set_ylabel('Reddit monthly submission size (bytes)')

ax2.grid(True, which="major", ls="-")
ax2.legend()
ax2.set_xlim(np.min(dt['date']),np.max(dt['date']))

ax1.set_xlabel('Year')
ax1.set_ylabel('Reddit monthly submission size (bytes)')

ax1.set_yscale('log')
ax1.grid(True, which="major", ls="-")
ax1.legend()
ax1.set_xlim(np.min(dt['date']),np.max(dt['date']))

fig.tight_layout()

ax1.grid('major', color='#F2F6F6', zorder=0)
ax2.grid('major', color='#F2F6F6', zorder=0)
plt.setp(ax1.spines.values(), color='#CCD8D9')
plt.setp(ax2.spines.values(), color='#CCD8D9')
ax1.tick_params(axis='both', which='both', color='#CCD8D9')
ax2.tick_params(axis='both', which='both', color='#CCD8D9')
fig.savefig('results/reddit.pdf')
plt.show()



