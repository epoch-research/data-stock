"""# Toy model of bottleneck"""

from matplotlib import pyplot as plt
import numpy as np

#A = 406.4
#B = 410.7
#a = 0.34
#b = 0.28

A = 514
B = 2115.2
a = 0.35
b = 0.37


# Loss as a function of C (compute spend during training as compute-optimally as possible)
# and D (maximum usable dataset size)
def RL(C, D, A=A, B=B, a=a, b=b):
    G = (a*A/b/B)**(1/(a+b))
    Nopt = G*(C/6)**(b/(a+b))
    Dopt = (C/6)**(a/(a+b)) / G
    return (
            (Dopt < D) * (A*(Nopt)**(-a) + B*(Dopt)**(-b))
            +
            (Dopt >= D) * (A*(C/6/D)**-a + B*D**-b))

def dopt(C, I=0, A=A, B=B, a=a, b=b):
    G = (a*A/b/B)**(1/(a+b))
    return (C/6)**(a/(a+b)) / G

# Returns to x% more compute and data
def ret_C(C,D, eps=0.00001):
    return (RL((1+eps)*C, D) - RL(C,D))/eps

def ret_D(C,D, eps=0.00001):
    return (RL(C, (1+eps)*D) - RL(C,D))/eps

C = np.logspace(15,25,1000, base=10)
D = np.logspace(8,13,1000, base=10)

x,y = np.meshgrid(C,D)
z = ret_D(x,y)/ret_C(x,y)

idxs = np.where((y < dopt(x)).flatten())

plt.scatter((np.sqrt(x)/y).flatten()[idxs], z.flatten()[idxs])
plt.axvline((6/20)**0.5)
plt.xlabel('sqrt(C)/D')
plt.ylabel('Returns to D over returns to C')
plt.xscale('log')
plt.yscale('log')
plt.show()

fig = plt.figure(figsize=(5,3))
ax = fig.subplots(1)

def loss(C, D, A=A, B=B, a=a, b=b):
    return A*(C/6/D)**(-a) + B*(D)**(-b)

def optscal(I, D, A=A, a=a, B=B, b=b):
    N = ((6*D**(-b) + 2*I*D**(-b-1)) * b*B/A/a/6)**(-1/a)
    return 6*N*D

C = np.logspace(22,30,1000, base=10)
D = np.logspace(11,15,1000, base=10)

x,y = np.meshgrid(C,D)
z = loss(x,y).clip(None, 10.82)

levels = np.logspace(np.log(0.13), np.log(10), 20)

import matplotlib as mpl
norm = mpl.colors.LogNorm(levels.min()*32, levels.max()/2)
cm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps['viridis'])
lcolors = [mpl.colormaps['viridis'](norm(x)) for x in reversed(levels)]

#plt.contour(x,y,z, levels, colors=lcolors, linestyles='dotted', linewidths=0.8)

#plt.plot(C, Dopt, color='red')

#Co = 6*(b*(b-1)*B/a/(a-1)/A)**(1/(1-a)) *D**((1-b)/(1-a)+1)
#plt.plot(Co, D)

#for k in [1e13, 1e14, 1e15]:
#  I=k
#  CI = optscal(I, D)
#  plt.plot(CI,D, color='blue')

#plt.plot(6*No*Do, Do)

plt.axhline(1e14, xmax=0.62, linestyle='--', color='gray')
plt.axvline(1e27, ymax=0.75, linestyle='--', color='gray')

n = np.logspace(8,14, 1000, base=10)
d = np.logspace(8,15, 1000, base=10)

N, D = np.meshgrid(n,d)

L = A*N**-a + B*D**-b

hs = [0.5]

I0 = 1e13

rs = [1,2,4,8]  # example value
costs = [np.log(
      a*A*N**(-a-1) / (b*B*D**(-b-1))) -
     (
         np.log(6*D + 2*I0/(0.1**-r)/(1e12**-h)*((1-h)*N**(-h)*L**-r + r*L**(-r-1) * a*A*N**(-a-h))) -
         np.log(6*N + 2*I0*N**(1-h)/(0.1**-r)/(1e12**-h)*(r*L**(-r-1) * b*B*D**(-b-1)))
    ) for h in hs for r in rs ]

idxs = [np.isclose(cost, 0, rtol=1e-3, atol=1e-3) for cost in costs]
Ns = [N[idx] for idx in idxs]
Ds = [D[idx] for idx in idxs]

for i, (N, D, (r,h)) in enumerate(zip(Ns, Ds, [(r,alpha) for alpha in hs for r in rs])):
  idx = (6*N*D < 1e27) * (D < 1e14)
  plt.plot((6*N*D)[idx],D[idx], label=f'r={r},h={h}', color=colors[1+i%(len(colors)-1)])
  plt.plot((6*N*D)[~idx],D[~idx], color=colors[1+i%(len(colors)-1)], linestyle=':')

Dopt = dopt(C)
idx = C < 1e27
plt.plot(C[idx], Dopt[idx], color=colors[0], label='Optimal')
plt.plot(C[~idx], Dopt[~idx], color=colors[0], linestyle=':')

plt.xlabel('Training compute (FLOP)')
plt.ylabel('Data (tokens)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower right")
plt.ylim(1e11,1e15)
plt.xlim(1e22,1e30)
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
fig.savefig('opt_model.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

fig = plt.figure(figsize=(5,3))
ax = fig.subplots(1)

C = np.logspace(22,30,1000, base=10)
D = np.logspace(11,15,1000, base=10)

Dopt = dopt(C)
idx = C < 1e27
plt.plot(C[idx], Dopt[idx], color=colors[0], label='Optimal')
plt.plot(C[~idx], Dopt[~idx], color=colors[0], linestyle=':')

plt.axhline(1e14, xmax=0.62, linestyle='--', color='gray')
plt.axvline(1e27, ymax=0.75, linestyle='--', color='gray')

n = np.logspace(7,14, 1000, base=10)
d = np.logspace(7,15, 1000, base=10)

N, D = np.meshgrid(n,d)

L = A*N**-a + B*D**-b

hs = [1, 1.1]

I0 = 3e13

rs = [2,4]  # example value
costs = [np.log(
      -h/N + r/L*a*A*N**(-a-1) / (r/L*b*B*D**(-b-1)) /
     (
         (6*D + 2*I0*(N/1e12)**-h*(L/0.1)**-r*(1-h+r/L*a*A*N**(-a))) /
         (6*N + 2*I0*(N/1e12)**-h*(L/0.1)**-r*(N*r/L*b*B*D**(-b-1)))
    )) for h in hs for r in rs ]

idxs = [np.isclose(cost, 0, rtol=1e-2, atol=1e-2) for cost in costs]
Ns = [N[idx] for idx in idxs]
Ds = [D[idx] for idx in idxs]

for i, (N, D, (r,h)) in enumerate(zip(Ns, Ds, [(r,alpha) for alpha in hs for r in rs])):
  idx = (6*N*D < 1e27) * (D < 1e14)
  plt.plot((6*N*D)[idx],D[idx], label=f'r={r},h={h}', color=colors[1+i%(len(colors)-1)])
  plt.plot((6*N*D)[~idx],D[~idx], color=colors[1+i%(len(colors)-1)], linestyle=':')


plt.xlabel('Training compute (FLOP)')
plt.ylabel('Data (tokens)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylim(1e11,1e15)
plt.xlim(1e22,1e30)
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
fig.savefig('opt_model.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

fig = plt.figure(figsize=(5,3))
ax = fig.subplots(1)

Dopt = dopt(C)
idx = C < 1e27
plt.plot(C[idx], (Dopt/(C/6/Dopt))[idx], color=colors[0], label='Optimal')
plt.plot(C[~idx], (Dopt/(C/6/Dopt))[~idx], color=colors[0], linestyle=':')

plt.plot(C[idx], 1e14/(C/6/1e14)[idx], linestyle='--', color='gray')
plt.axvline(1e27, ymax=0.45, linestyle='--', color='gray')

for i, (N, D, (r,alpha)) in enumerate(zip(Ns, Ds, [(r,alpha) for alpha in alphas for r in rs])):
  idx = (6*N*D < 1e27) * (D < 1e14)
  plt.plot((6*N*D)[idx],(D/N)[idx], label=f'r={r},h={alpha}', color=colors[1+i%(len(colors)-1)])
  plt.plot((6*N*D)[~idx],(D/N)[~idx], color=colors[1+i%(len(colors)-1)], linestyle=':')


plt.xlabel('Training compute (FLOP)')
plt.ylabel('Data / paramter ratio')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylim(1,1e4)
plt.xlim(1e22,1e30)
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
fig.savefig('opt_model_ratio.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

for i, (N, D, (r,h)) in enumerate(zip(Ns, Ds, [(r,h) for h in hs for r in rs])):
  idx = (6*N*D < 1e27) * (D < 1e14)
  plt.plot((6*N*D),2*N*I0*((A*N**-a + B*D**-b)/0.1)**(-r)*(N/1e12)**-h, label=f'r={r},h={h}', color=colors[1+i%(len(colors)-1)])
  #plt.scatter((6*N*D),(A*N**-a + B*D**-b), label=f'r={r},h={h}', color=colors[1+i%(len(colors)-1)])

plt.xscale('log')
plt.yscale('log')
plt.ylabel('Total inference compute (FLOP)')
plt.xlabel('Training compute (FLOP)')
plt.grid()
#plt.ylim(1e12,1e15)
plt.ylim(1e22,1e28)
plt.xlim(1e22,1e30)
plt.legend(loc="lower right")
plt.show()

from scipy.optimize import minimize

def objective(lN,lD):
    N = np.exp(lN)
    D = np.exp(lD)
    return A * N**(-a) + B * D**(-b)

def constraint(lN, lD, r, h, I0, A=A, a=a, B=B, b=b):
    N = np.exp(lN)
    D = np.exp(lD)
    L = objective(lN,lD)
    return np.log(a*A*N**(-a-1) /b/B/D**(-b-1) / ((6*D + 2*I0*((1-h)*N**(-h)*L**-r + r*L**(-r-1) * a*A*N**(-a-h))) / (6*N + 2*I0*N**(1-h)*(r*L**(-r-1) * b*B*D**(-b-1)))))

D = np.log(np.logspace(13,16,10,base=10))

N = np.zeros_like(D)

for i,d in enumerate(D):
  con = {'type': 'eq', 'fun': lambda x: constraint(x, d, 4, 1, 1e22)}
  x0 = [8]
  result = minimize(lambda x: objective(x, d), x0, method='SLSQP', constraints=con)
  N[i] = result.x

n = np.linspace(10,15,100).reshape(-1,1)
d = np.linspace(10,15,100).reshape(1,-1)
plt.imshow(constraint(n,d,2,1,1e22))
plt.colorbar()
plt.show()

L = A*np.exp(N)**-a + B*np.exp(D)**-b
plt.scatter(6*np.exp(N)*np.exp(D), 1e22*L**-4 * np.exp(N)**-1)
plt.xscale('log')
plt.yscale('log')
plt.show()

print(np.log10(np.exp(N)))
print(np.log10(np.exp(D)))
c = np.logspace(18,27, 100, base=10)
ddo = dopt(c)
plt.plot(c, ddo/(c/6/ddo))
plt.scatter(6*np.exp(N)*np.exp(D), np.exp(D-N))
plt.xscale('log')
plt.yscale('log')
plt.show()

def optscal(I, D, A=A, a=a, B=B, b=b):
    N = ((6*D**(-b) + 2*I*D**(-b-1)) * b*B/A/a/6)**(-1/a)
    return N

print(1e16/optscal(1e9, 1e16))

fig = plt.figure(figsize=(5,3))
ax = fig.subplots(1)

D = 3e14
comp = np.logspace(27,33,100,base=10)
plt.plot(comp, RL(comp, D), label="Undertraining (data bottleneck)", color=color_data)
plt.axvline(D*D/20*6, color='black')
plt.plot(comp, RL(comp, 1e30), label="Compute-optimal (no bottleneck)", color=color_comp)
plt.ylabel('Reducible loss')
plt.xlabel('Compute')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.yticks([0.006, 0.01, 0.02, 0.03, 0.04], ['0.006', '0.01', '0.02', '0.03', '0.04'])
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
fig.savefig('toy_model_undertraining.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

[((3e14/np.quantile(dataset_size_comp((t-2024)*365), 0.5))**2,t) for t in np.linspace(2024,2030,13)]

10**compute_proj.groupby('year')["comp"].mean().diff()

x = np.linspace(2024, 2034, 26)
comp = np.array([np.median(available_comp((d-2024)*365)) for d in x])
plt.plot(x, RL(10**comp, np.median(dataset_size(0))), label="Data bottleneck", color=color_data)
plt.plot(x, RL(10**comp, np.median(dataset_size(40*365))), label="No bottleneck", color=color_comp)
plt.ylabel('Reducible loss')
plt.xlabel('Year')
plt.yscale('log')
plt.yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], ['0.01', '0.02', '0.03', '0.04', '', '0.06', ''])
plt.legend()
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.show()

"""## Profit maximization"""

import numpy as np
from scipy.optimize import minimize

A = 514
B = 2115.2
a = 0.35
b = 0.37

I0 = 1e13
r=2
h=1.1

def profit(x):
    lN, lD, P = x
    N = 10**lN
    D = 10**lD
    L = A * N**-a + B * D**-b
    return -(I0 * (L/0.1)**-r * P**(1-h) - 6 * N * D - 2 * N * I0 * (L/0.1)**-r * P**-h)  # Negate for maximization

def demand(x):
    lN, lD, P = x
    N = 10**lN
    D = 10**lD
    L = A * N**-a + B * D**-b
    return I0*(L/0.1)**-r*P**-h

# Initial guesses
initial_guess = [9, 9, 1.5]  # Logarithmic scale initial guess

# Bounds for lN, lD, and P assuming some practical ranges
bounds = [(0, 15), (0, 15), (0, 1_000_000)]

# Optimization
result = minimize(profit, initial_guess, bounds=bounds, method='L-BFGS-B')

# Check results
if result.success:
    optimized_lN, optimized_lD, optimized_P = result.x
    print(f"Optimized lN: {optimized_lN}, lD: {optimized_lD}, P: {optimized_P}")
    print(f"Maximum profit: {-result.fun}")  # Negate back since we minimized
    print(f"Demand: {demand(result.x)}")
else:
    print("Optimization failed:", result.message)

import torch

A = torch.tensor(514)
B = torch.tensor(2115.2)
a = torch.tensor(0.35)
b = torch.tensor(0.37)

I0 = torch.tensor(1e13)
r = torch.tensor(2)
h = torch.tensor(1)

def profit(x, reg=True):
    lN, lD, lP = x
    L = torch.logsumexp(torch.stack((A.log() - a*lN, B.log() - b*lD),0),0)
    if reg:
      rg = torch.sigmoid(-1000*(np.log(6)+lN+lD-np.log(10)*25))
    else:
      rg = 1
    return ((I0.log() -r*(L - torch.tensor(0.1).log()) + (1-h)*lP).exp()
                             - (torch.tensor(6).log() + lN + lD).exp()
                             - (torch.tensor(2).log() + lN + I0.log() -r*(L - torch.tensor(0.1).log()) - h*lP).exp())*rg

def demand(x):
    lN, lD, lP = x
    N = lN.exp()
    D = lD.exp()
    L = A * N**-a + B * D**-b
    print(L.item())
    return I0*(L)**-r*(lP.exp())**-h

# Variables (initial values and requiring gradient)
lN = torch.tensor([20.0], requires_grad=True)
lD = torch.tensor([24.0], requires_grad=True)
lP = torch.tensor([2.0], requires_grad=True)

# Optimizer setup
optimizer = torch.optim.Adam([lN, lD, lP], lr=0.01)

# Optimization loop
for _ in range(10_000):  # Number of optimization steps
    optimizer.zero_grad()  # Clear gradients
    loss = -profit((lN, lD, lP))  # Negate to maximize
    loss.backward()  # Compute gradient
    optimizer.step()  # Update parameters
    with torch.no_grad():
      lN.clamp_(max=40)
      lD.clamp_(max=40)
      lP.clamp_(max=40)

print(f"Optimized lN: {lN.exp().item():.2e}, lD: {lD.exp().item():.2e}, P: {lP.exp().item():.2e}")
print(f"Maximum profit: {profit((lN, lD, lP), reg=False).exp().item():.2e}")
print(f"Demand: {demand((lN, lD, lP)).item():.2e}")
print(f"Revenue: {lP.exp().item()*demand((lN, lD, lP)).item():.2e}")
print(f"Training compute: {6*lN.exp().item()*lD.exp().item():.2e}")
print(f"Inference compute: {2*lN.exp().item()*demand((lN, lD, lP)).item():.2e}")

