from utils import *
import torch

A = 514
B = 2115.2
a = 0.35
b = 0.37

I0 = 1e15
r = 7
h = 2

def dopt(C, I=0, A=A, B=B, a=a, b=b):
    G = (a*A/b/B)**(1/(a+b))
    return (C/6)**(a/(a+b)) / G

def profit(x, A=a, a=a, B=B, b=b, I0=I0, r=r, h=h, reg=True, C=25):
    lN, lD, lP = x
    N = lN.exp()
    D = lD.exp()
    P = lP.exp()
    L = A*N**-a + B*D**-b
    if reg:
      rg = torch.sigmoid(-10000*(np.log(6)+lN+lD-np.log(10)*C))
    else:
      rg = 1
    return (I0*(L/0.1)**-r*(P/1e-6)**-h*(P - 2*N/1e18) - 6*N*D/1e18)#*rg
    #return ((I0.log() -r*(L - torch.tensor(0.1).log()) + (1-h)*lP).exp()
    #                         - (torch.tensor(6).log() + lN + lD).exp()
    #                         )#- (torch.tensor(2).log() + lN + I0.log() -r*(L - torch.tensor(0.1).log()) - h*lP).exp())*rg

def demand(x, A=A, a=a, B=B, b=b, I0=I0, r=r, h=h):
    lN, lD, lP = x
    N = lN.exp()
    D = lD.exp()
    P = lP.exp()
    L = A * N**-a + B * D**-b
    #print(L.item())
    return I0*(L/0.1)**-r*(P/1e-6)**-h

def solve(n0=20.0, d0=20.0, p0=-6.0, C=25, verbose=False, A=A, a=a, B=B, b=b, I0=I0, r=r, h=h):
    # Variables (initial values and requiring gradient)
    lN = torch.tensor([n0], requires_grad=True)
    lD = torch.tensor([d0], requires_grad=True)
    lP = torch.tensor([p0], requires_grad=True)

    tA = torch.tensor(A)
    ta = torch.tensor(a)
    tB = torch.tensor(B)
    tb = torch.tensor(b)

    tI0 = torch.tensor(I0)
    tr = torch.tensor(r)
    th = torch.tensor(h)

    # Optimizer setup
    optimizer = torch.optim.SGD([lN, lD, lP], lr=0.1)

    # Optimization loop
    for _ in range(2_000):  # Number of optimization steps
        optimizer.zero_grad()  # Clear gradients
        loss = -profit((lN, lD, lP), tA,ta,tB,tb,tI0,tr,th, C=C)  # Negate to maximize
        loss.backward()  # Compute gradient
        #print(lN.grad, lD.grad, lP.grad)
        torch.nn.utils.clip_grad_norm_([lN,lD,lP], max_norm=0.5, norm_type=1)
        #print(lN.grad, lD.grad, lP.grad)
        optimizer.step()  # Update parameters
        #print(lN, lD, lP)
        #break
        with torch.no_grad():
            rat = (lN + lD) / (np.log(10)*C-np.log(6))
            if rat > 1:
              lN /= rat
              lD /= rat
            rat = (lN + demand((lN, lD, lP), tA,ta,tB,tb,tI0,tr,th).log()) / (np.log(10)*C-np.log(2) + np.log(1000))
            if rat > 1:
              lN /= rat
          #lN.clamp_(max=30)
          #lD.clamp_(max=30)
          #lP.clamp_(max=10)


    if verbose:
        N = lN.exp()
        D = lD.exp()
        P = lP.exp()
        L = A * N**-a + B * D**-b
        print(L)
        print(lN-torch.tensor([n0]), lD-torch.tensor([d0]), lP-torch.tensor([p0]))
        print(f"Optimized lN: {lN.exp().item():.2e}, lD: {lD.exp().item():.2e}, P: {lP.exp().item():.2e}")
        print(f"Maximum profit: {profit((lN, lD, lP), reg=False).item():.2e}")
        print(f"Demand: {demand((lN, lD, lP), tA,ta,tB,tb,tI0,tr,th).item():.2e}")
        print(f"Revenue: {lP.exp().item()*demand((lN, lD, lP), tA,ta,tB,tb,tI0,tr,th).item():.2e}")
        print(f"Training compute: {6*lN.exp().item()*lD.exp().item():.2e}")
        print(f"Inference compute: {2*lN.exp().item()*demand((lN, lD, lP), tA,ta,tB,tb,tI0,tr,th).item():.2e}")
    return lN.exp().item(), lD.exp().item(), lP.exp().item()


def optimal_scaling(C, A=A, a=a, B=B, b=b, I0=I0, r=r, h=h):
    ns = []
    ds = []
    ps = []
    for c in C:
        n, d, p = solve(np.log((c/6)**0.5/5), np.log((c/6)**0.5*5), -12., C=np.log10(c), A=A, a=a, B=B, b=b, I0=I0, r=r, h=h)
        ns.append(n)
        ds.append(d)
        ps.append(p)

    return np.array(ns), np.array(ds), np.array(ps)


def numerical_optscal_data(comp, slcomp, sldata):
    return ((slcomp[-1]-comp)*sldata[0] + (comp-slcomp[0])*sldata[-1])/(slcomp[-1]-slcomp[0])

def interpolate(x, y, n):
    nx, ny = [], []
    for i in range(len(x)-1):
      nx.extend([np.exp((n-j)/n*np.log(x[i]) + j/n*np.log(x[i+1])) for j in range(n)])
      ny.extend([np.exp((n-j)/n*np.log(y[i]) + j/n*np.log(y[i+1])) for j in range(n)])
    return np.array(nx), np.array(ny)


def get_optimal_policies():
    C = np.logspace(22,31,10,base=10)

    ns1, ds1, ps1 = optimal_scaling(C, I0=1e13, r=7, h=1.1)
    ns2, ds2, ps2 = optimal_scaling(C, I0=1e15, r=7, h=2)
    ns3, ds3, ps3 = optimal_scaling(C, I0=1e16, r=10, h=2.4)

    return C, ns1, ds1, ps1, ns2, ds2, ps2, ns3, ds3, ps3

if __name__ == '__main__':
    max_data = 1e14
    max_compute = 5e25

    C, ns1, ds1, ps1, ns2, ds2, ps2, ns3, ds3, ps3 = get_optimal_policies()

    fig = plt.figure(figsize=(5,6))
    ax1, ax2 = fig.subplots(2, sharex=True)

    ax1.axhline(max_data, xmax=0.46, linestyle='--', color='gray')
    ax1.axvline(max_compute, ymax=0.75, linestyle='--', color='gray')
    ax2.axvline(max_compute, ymax=0.76, linestyle='--', color='gray')

    Dopt = dopt(C)
    Ci, Dopti = interpolate(C, Dopt, 20)
    idx = Ci <= max_compute
    ax1.plot(Ci[idx], Dopti[idx], color=colors[0], label='Optimal')
    ax1.plot(Ci[~idx], Dopti[~idx], color=colors[0], linestyle=':')
    ax2.plot(Ci[idx], (max_data/(Ci/6/max_data))[idx], linestyle='--', color='gray')
    ax2.plot(Ci[idx], (Dopti/(Ci/6/Dopti))[idx], color=colors[0], label='Optimal')
    ax2.plot(Ci[~idx], (Dopti/(Ci/6/Dopti))[~idx], color=colors[0], linestyle=':')

    cs1 = np.array(ds1)*np.array(ns1)*6
    csi1, dsi1 = interpolate(cs1, ds1, 20)
    idx = csi1 <= max_compute
    ax1.plot(csi1[idx], dsi1[idx], color=colors[2], label="r=7,h=1.1")
    ax1.plot(csi1[~idx], dsi1[~idx], color=colors[2], linestyle=':')
    ax2.plot(csi1[idx], (dsi1/(csi1/6/dsi1))[idx], color=colors[2], label="r=7,h=1.1")
    ax2.plot(csi1[~idx], (dsi1/(csi1/6/dsi1))[~idx], color=colors[2], linestyle=':')

    cs2 = np.array(ds2)*np.array(ns2)*6
    csi2, dsi2 = interpolate(cs2, ds2, 20)
    idx = csi2 <= max_compute
    ax1.plot(csi2[idx], dsi2[idx], color=colors[3], label="r=7,h=2")
    ax1.plot(csi2[~idx], dsi2[~idx], color=colors[3], linestyle=':')
    ax2.plot(csi2[idx], (dsi2/(csi2/6/dsi2))[idx], color=colors[3], label="r=7,h=2")
    ax2.plot(csi2[~idx], (dsi2/(csi2/6/dsi2))[~idx], color=colors[3], linestyle=':')

    cs3 = np.array(ds3)*np.array(ns3)*6
    csi3, dsi3 = interpolate(cs3, ds3, 20)
    idx = dsi3 <= max_data
    ax1.plot(csi3[idx], dsi3[idx], color=colors[5], label="r=10,h=2.3")
    ax1.plot(csi3[~idx], dsi3[~idx], color=colors[5], linestyle=':')
    ax2.plot(csi3[idx], (dsi3/(csi3/6/dsi3))[idx], color=colors[5], label="r=10,h=2.3")
    ax2.plot(csi3[~idx], (dsi3/(csi3/6/dsi3))[~idx], color=colors[5], linestyle=':')

    ax1.set_ylabel('Data (tokens)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([])
    ax1.legend(loc="lower right")
    ax1.set_ylim(1e11,1e15)
    ax1.set_xlim(1e22,1e30)
    ax1.grid('major', color='#F2F6F6', zorder=0)
    plt.setp(ax1.spines.values(), color='#CCD8D9')
    ax1.tick_params(axis='both', which='both', color='#CCD8D9')
    ax1.tick_params(axis='x', which='both', bottom=False)

    ax2.set_xlabel('Training compute (FLOP)')
    ax2.set_ylabel('Data / paramter ratio')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_yticks([1e0,1e1,1e2,1e3])
    ax2.legend()
    ax2.set_ylim(1,1e4)
    ax2.set_xlim(1e22,1e30)
    ax2.grid('major', color='#F2F6F6', zorder=0)
    plt.setp(ax2.spines.values(), color='#CCD8D9')
    ax2.tick_params(axis='both', which='both', color='#CCD8D9')

    plt.tight_layout()
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0.05, wspace = 0)
    fig.savefig('results/opt_model.pdf', bbox_inches = 'tight', pad_inches=0)
    plt.show()



    fig = plt.figure(figsize=(5,3))
    ax = fig.subplots(1)

    plt.axvline(1e27, ymax=0.43, linestyle='--', color='gray')

    idx = Ci <= 1e27
    plt.plot(Ci[idx], (1e14/(Ci/6/1e14))[idx], linestyle='--', color='gray')
    plt.plot(Ci[idx], (Dopti/(Ci/6/Dopti))[idx], color=colors[0], label='Optimal')
    plt.plot(Ci[~idx], (Dopti/(Ci/6/Dopti))[~idx], color=colors[0], linestyle=':')

    idx = csi1 <= 1e27
    plt.plot(csi1[idx], (dsi1/(csi1/6/dsi1))[idx], color=colors[2], label="r=7,h=1.1")
    plt.plot(csi1[~idx], (dsi1/(csi1/6/dsi1))[~idx], color=colors[2], linestyle=':')

    idx = dsi2 <= 1e14
    plt.plot(csi2[idx], (dsi2/(csi2/6/dsi2))[idx], color=colors[3], label="r=7,h=2")
    plt.plot(csi2[~idx], (dsi2/(csi2/6/dsi2))[~idx], color=colors[3], linestyle=':')

    idx = dsi3 <= 1e14
    plt.plot(csi3[idx], (dsi3/(csi3/6/dsi3))[idx], color=colors[5], label="r=10,h=2.3")
    plt.plot(csi3[~idx], (dsi3/(csi3/6/dsi3))[~idx], color=colors[5], linestyle=':')

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
    fig.savefig('results/opt_model_ratio.pdf', bbox_inches = 'tight', pad_inches=0)
    plt.show()
