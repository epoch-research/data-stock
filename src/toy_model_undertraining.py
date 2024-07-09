"""# Toy model of bottleneck"""

from utils import *

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



if __name__ == '__main__':
    fig = plt.figure(figsize=(6,3), dpi=300)
    ax = fig.subplots(1)

    D = 3e14
    comp = np.logspace(27,33,100,base=10)
    plt.plot(comp, RL(comp, D), label="Undertraining (data bottleneck)", color=historical_projection_color, zorder=50)
    plt.axhline(B*D**-b, xmin=0.6, color=color_data, linestyle='--')
    plt.plot(comp, RL(comp, 1e30), label="Compute-optimal (no bottleneck)", color=compute_projection_color, zorder=50)
    plt.scatter((D*D/20*6), (RL(D*D/20*6, D)), color='black', marker='x', s=50, zorder=100, label='Start of bottleneck')
    plt.scatter((3e30), (B*D**-b), color='black', marker='+', s=80, zorder=100, label='Compute-optimal equivalent\nof the plateau level')
    plt.axvline(D*D/20*6, ymax=0.75, color=color_data, linestyle=':')
    plt.axvline(3e30, ymax=0.43, color=color_data, linestyle=':')
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
    plt.margins(0,0.015)
    fig.savefig('results/toy_model_undertraining.jpg', bbox_inches = 'tight', pad_inches=0)
    plt.show()
