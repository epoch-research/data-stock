### Dataset size projections ###

from utils import *
from math import floor, ceil

def get_dataset_size_projection():
    df = pd.read_csv('data/dataset_sizes.csv')
    df.date = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df = df[df.date > dt.datetime(2010, 1, 1)]
    df.dataset = df.dataset * 4 / 5 # convert to tokens

    slopes = []
    intercepts = []
    for i in tqdm(range(10000)):
        draws = np.random.randint(0, df.shape[0], df.shape[0])
        data = df.iloc[draws, :]
        res = linregress((data.date-dt.datetime(2024, 1, 1)).apply(lambda x: x.days), np.log10(data.dataset))
        slopes.append(res.slope)
        intercepts.append(res.intercept)

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    print(f"\nMedian: {365*np.quantile(slopes, 0.5):.2f}")
    print(f"95% CI: [{365*np.quantile(slopes, 0.025):.2f},{365*np.quantile(slopes, 0.975):.2f}]")

    yearly_growth = sq.to(10**np.quantile(slopes, 0.025), 10**np.quantile(slopes, 0.975), credibility=95)
    current_largest_dataset = sq.to(3e12, 13e12) #sq.to(10**np.quantile(intercepts, 0.05), 10**np.quantile(intercepts, 0.95))

    def dataset_size_hist(y):
        return yearly_growth**y * current_largest_dataset


    compute_proj = pd.read_csv('data/compute_projection.csv', names=["idx", "comp", "year"])

    _years = compute_proj.year.unique()
    _years.sort()

    def available_comp(y):
        y = y/365+2024
        idx = np.searchsorted(_years, y)
        before = _years[idx-1]
        after = _years[idx]
        comp = compute_proj.comp[compute_proj.year == before].to_numpy()*(after-y)/(after-before) + compute_proj.comp[compute_proj.year == after].to_numpy()*(y-before)/(after-before)
        return comp

    comp_2024 = sq.to(10**np.quantile(available_comp(0), 0.05), 10**np.quantile(available_comp(0), 0.95))

    slopesc = []
    interceptsc = []
    _data = df.dropna(axis=0, subset=["comp"])
    for i in tqdm(range(10000)):
      draws = np.random.randint(0, _data.shape[0], _data.shape[0])
      data = _data.iloc[draws, :]
      res = linregress((data.date-dt.datetime(2024, 1, 1)).apply(lambda x: x.days), np.log10(data.comp))
      slopesc.append(res.slope)
      interceptsc.append(res.intercept)

    slopesc = np.array(slopesc)
    interceptsc = np.array(interceptsc)
    print(f"\nMedian: {365*np.quantile(slopesc, 0.5):.2f}")
    print(f"90% CI: [{365*np.quantile(slopesc, 0.05):.2f},{365*np.quantile(slopesc, 0.95):.2f}]")

    yearly_comp_growth = sq.to(10**np.quantile(slopesc, 0.05), 10**np.quantile(slopesc, 0.95))
    current_largest_comp = sq.to(1e25, 1e26)

    def hist_comp(y):
        return yearly_comp_growth**y * comp_2024

    def dataset_size_comp(y):
        if y >= 0:
          comp = 10**available_comp(y)
        else:
          comp = hist_comp(y) @ 10_000
        return 20*(comp/6/20)**0.5

    def dataset_size(y):
        return np.concatenate((dataset_size_hist(y) @ 10_000, dataset_size_comp(y)))

    return dataset_size, dataset_size_hist, dataset_size_comp

if __name__ == '__main__':
    import datetime as dt

    df = pd.read_csv('data/dataset_sizes.csv')
    df.date = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df = df[df.date > dt.datetime(2010, 1, 1)]
    df.dataset = df.dataset * 4 / 5 # convert to tokens

    dataset_size, dataset_size_hist, dataset_size_comp = get_dataset_size()

    fig = plt.figure(figsize=(5,3))
    ax = fig.subplots(1)

    x = pd.date_range(start='2010-01-01',
                      end='2040-01-01',
                      periods=100)

    plt.scatter(df.date, df.dataset, color=model_fill_color, edgecolors=model_fill_color, alpha=model_fill_alpha, zorder=10)

    hist_samples = [dataset_size_hist((y-dt.datetime(2024, 1, 1)).days) @ 100_000 for y in x]
    q05 = [np.quantile(x, 0.05) for x in hist_samples]
    q50 = [np.quantile(x, 0.50) for x in hist_samples]
    q95 = [np.quantile(x, 0.95) for x in hist_samples]
    plt.fill_between(x, q05, q95, color=historical_projection_color, alpha=historical_projection_ci_alpha, zorder=20)
    plt.plot(x, q50, color=historical_projection_color, label="Historical projection", zorder=20)


    comp_samples = [dataset_size_comp((y-dt.datetime(2024, 1, 1)).days) for y in x]
    q05c = [np.quantile(x, 0.05) for x in comp_samples]
    q50c = [np.quantile(x, 0.50) for x in comp_samples]
    q95c = [np.quantile(x, 0.95) for x in comp_samples]
    plt.fill_between(x, q05c, q95c, color=compute_projection_color, alpha=compute_projection_ci_alpha, zorder=20)
    plt.plot(x, q50c, color=compute_projection_color, label="Compute-based projection", zorder=20)

    plt.text(datetime(2011,1,1), 1e12, "Word2Vec")
    plt.plot([datetime(2013,6,1), datetime(2013,10,16)], [6e11, 3.3e10], color='black', linewidth=0.7, zorder=30)
    plt.text(datetime(2017,1,1), 1e6, "Transformer")
    plt.plot([datetime(2018,1,1), datetime(2017,7,1)], [5e6, 8e8], color='black', linewidth=0.7, zorder=30)
    plt.text(datetime(2019,1,1), 2e13, "GPT-3")
    plt.plot([datetime(2020,1,1), datetime(2020,5,28)], [1.3e13, 3.74e11], color='black', linewidth=0.7, zorder=30)
    plt.text(datetime(2025,12,1), 5e11, "Llama 3-70B")
    plt.plot([datetime(2025,11,1), datetime(2024,5,1)], [1e12, 1e13], color='black', linewidth=0.7, zorder=30)

    plt.legend()
    plt.yscale('log')
    plt.ylabel('Tokens')
    plt.xlabel('Year')
    plt.ylim(1e5,1e18)
    plt.grid('major', color='#F2F6F6', zorder=0)
    plt.setp(ax.spines.values(), color='#CCD8D9')
    plt.tick_params(axis='both', which='both', color='#CCD8D9')
    plt.tight_layout()
    plt.margins(0,0)
    fig.savefig('dataset_sizes.pdf', bbox_inches = 'tight', pad_inches=0)
    plt.show()
