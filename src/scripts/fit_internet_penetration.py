from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

data = pd.read_csv('share-of-individuals-using-the-internet.csv')
xdata = data['year']
ydata = data['internet_pen']

from matplotlib import pyplot as plt

def sigmoid(x, x0, k, L=100, b=0):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

p0 = [np.median(xdata),1] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')

plt.plot(xdata, ydata, label='real')
plt.plot(xdata, [sigmoid(x, *popt) for x in xdata], label='fit')

plt.legend()
plt.show()

print(popt)
print(pcov)
