from utils import *
from indexed_web import stock_indexed_web
from internet_users import stock_internet_users

def quality_adjust(x):
    # From the RefinedWeb paper, about 25% of the English data can be used
    # after cleaning.
    return x * sq.to(0.1, 0.4, credibility=95)

def max_repetition_scaling(x):
    # Max amount of effective data increase from repetition
    # Endpoints from https://arxiv.org/pdf/2305.16264.pdf, Table 1
    R = sq.to(2, 5, credibility=95)
    return x*R

_cache = dict()
def aggregate_raw(y):
    if y not in _cache:
      _cache[y] = stock_indexed_web(y) #sq.mixture((stock_cc(y), stock_internet_users(y)), relative_weights=(2,1))
    return _cache[y]

def aggregate_qual(y):
    return quality_adjust(aggregate_raw(y))

def aggregate(y):
    return max_repetition_scaling(aggregate_qual(y))

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.subplots(1)

    x = np.linspace(2024, 2040, 17)

    stock_raw = [aggregate_raw(y) @ 10_000 for y in x]
    sq95r = [np.quantile(s, 0.95) for s in stock_raw]
    sq50r = [np.quantile(s, 0.50) for s in stock_raw]
    sq05r = [np.quantile(s, 0.05) for s in stock_raw]

    stock_qual = [aggregate_qual(y) @ 10_000 for y in x]
    sq95q = [np.quantile(s, 0.95) for s in stock_qual]
    sq50q = [np.quantile(s, 0.50) for s in stock_qual]
    sq05q = [np.quantile(s, 0.05) for s in stock_qual]

    stock = [aggregate(y) @ 10_000 for y in x]
    sq95 = [np.quantile(s, 0.95) for s in stock]
    sq50 = [np.quantile(s, 0.50) for s in stock]
    sq05 = [np.quantile(s, 0.05) for s in stock]

    plt.plot(x, sq50r, label='Raw stock', color = colors[0])
    #plt.plot(x, sq05r, linestyle=':', color=colors[0])
    #plt.plot(x, sq95r, linestyle=':', color=colors[0])

    plt.plot(x, sq50q, label='Quality-adjusted stock', color=colors[2])
    #plt.plot(x, sq05q, linestyle=':', color=colors[2])
    #plt.plot(x, sq95q, linestyle=':', color=colors[2])

    plt.plot(x, sq50, label='Repetition-adjusted stock', color=colors[4])
    #plt.plot(x, sq05, linestyle=':', color=colors[4])
    #plt.plot(x, sq95, linestyle=':', color=colors[4])

    plt.annotate('', xy=(2026,3e14), xytext=(2026,7e13), arrowprops=dict(arrowstyle='<-'))
    plt.text(2026.5, 1.1e14, "Quality\nadjustment")
    plt.annotate('', xy=(2036,8e14), xytext=(2036,1.1e14), arrowprops=dict(arrowstyle='->'))
    plt.text(2036.4, 2e14, "Repetition\nadjustment")

    plt.legend(loc="upper left")
    plt.yscale('log')
    plt.xlim(2024, 2040)
    plt.ylabel('Effective stock (number of tokens)')
    plt.xlabel('Year')
    plt.grid('major', color='#F2F6F6', zorder=0)
    plt.setp(ax.spines.values(), color='#CCD8D9')
    plt.tick_params(axis='both', which='both', color='#CCD8D9')
    plt.tight_layout()
    fig.savefig('results/stocks.pdf')
    plt.show()

    print(f"""\
    {np.quantile(stock_indexed_web(2024) @ 100_000, 0.5) / 1e12:.1e} [95%: \
    {np.quantile(stock_indexed_web(2024) @ 100_000, 0.025) / 1e12:.1e}, \
    {np.quantile(stock_indexed_web(2024) @ 100_000, 0.975) / 1e12:.1e}]\
    """
    )
    print(f"""\
    {np.quantile(stock_internet_users(2024) @ 100_000, 0.5) / 1e12:.1e} [95%: \
    {np.quantile(stock_internet_users(2024) @ 100_000, 0.025) / 1e12:.1e}, \
    {np.quantile(stock_internet_users(2024) @ 100_000, 0.975) / 1e12:.1e}]\
    """
    )
    print(f"""\
    {np.quantile(aggregate_qual(2024) @ 100_000, 0.5) / 1e12:.1e} [95%: \
    {np.quantile(aggregate_qual(2024) @ 100_000, 0.025) / 1e12:.1e}, \
    {np.quantile(aggregate_qual(2024) @ 100_000, 0.975) / 1e12:.1e}]\
    """
    )
    print(f"""\
    {np.quantile(aggregate(2024) @ 100_000, 0.5) / 1e12:.1e} [95%: \
    {np.quantile(aggregate(2024) @ 100_000, 0.025) / 1e12:.1e}, \
    {np.quantile(aggregate(2024) @ 100_000, 0.975) / 1e12:.1e}]\
    """
    )
