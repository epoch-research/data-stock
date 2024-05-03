from utils import *
from indexed_web import *



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
fig.savefig('stocks.pdf')
plt.show()

print(f"""\
{np.quantile(stock_indexed_web(2024) @ 100_000, 0.5) / 1e12:.1e}[95%: \
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

"""# Full projections"""

def projection(y):
    stock = aggregate(y) @ 10_000
    stock_quantiles = [np.quantile(stock, q) for q in np.linspace(0,1,101)]
    dataset = []
    run_out = []
    for i in range(100):
        samples = sq.rclip(dataset_size_hist((y-2024)*365), stock_quantiles[i+1]) @ 1000
        dataset.append(samples)
        run_out.append(np.isclose(samples, stock_quantiles[i+1]))
    return np.concatenate(dataset), np.concatenate(run_out)

def projection_comp(y):
    stock = aggregate(y) @ 10_000
    stock_quantiles = [np.quantile(stock, q) for q in np.linspace(0,1,101)]
    dataset = []
    run_out = []
    for i in range(100):
        samples = dataset_size_comp((y-2024)*365).clip(0, stock_quantiles[i+1])
        dataset.append(samples)
        run_out.append(np.isclose(samples, stock_quantiles[i+1]))
    return np.concatenate(dataset), np.concatenate(run_out)

def aggregate_projection(y):
    stock = aggregate(y) @ 10_000
    stock_quantiles = [np.quantile(stock, q) for q in np.linspace(0,1,101)]
    dataset = []
    run_out = []
    for i in range(100):
        samples = dataset_size((y-2024)*365).clip(0, stock_quantiles[i+1])
        dataset.append(samples)
        run_out.append(np.isclose(samples, stock_quantiles[i+1]))
    return np.concatenate(dataset), np.concatenate(run_out)

x = np.linspace(2020,2041,100)#list(range(2020, 2042))

stock = [aggregate(y) @ 10_000 for y in x]
sq95 = [np.quantile(s, 0.95) for s in stock]
sq50 = [np.quantile(s, 0.50) for s in stock]
sq05 = [np.quantile(s, 0.05) for s in stock]

#samples, samples_run_out = zip(*[projection(y) for y in x])
#q95 = [np.quantile(s, 0.95) for s in samples]
#q50 = [np.quantile(s, 0.50) for s in samples]
#q05 = [np.quantile(s, 0.05) for s in samples]

#samples_comp, samples_run_out_comp = zip(*[projection_comp(y) for y in x])
#q95c = [np.quantile(s, 0.95) for s in samples_comp]
#q50c = [np.quantile(s, 0.50) for s in samples_comp]
#q05c = [np.quantile(s, 0.05) for s in samples_comp]

samples_agg, samples_run_out_agg = zip(*[aggregate_projection(y) for y in x])
q95a = [np.quantile(s, 0.95) for s in samples_agg]
q50a = [np.quantile(s, 0.50) for s in samples_agg]
q05a = [np.quantile(s, 0.05) for s in samples_agg]

fig = plt.figure(figsize=(5,3))
ax = fig.subplots(1)

plt.plot(x, sq50, label='Stock of data', color='black', linestyle='dotted')
plt.plot(x, sq95, linestyle=(0, (1,10)), color='black')
plt.plot(x, sq05, linestyle=(0, (1,10)), color='black')

#run_out_probs = [s.mean() for s in samples_run_out]
#median_runout_year = x[np.searchsorted(run_out_probs, 0.5)]
#q95_runout_year = x[np.searchsorted(run_out_probs, 0.95)]
#q05_runout_year = x[np.searchsorted(run_out_probs, 0.05)]

#run_out_probs_comp = [s.mean() for s in samples_run_out_comp]
#median_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.5)]
#q95_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.95)]
#q05_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.05)]

run_out_probs_agg = [s.mean() for s in samples_run_out_agg]
median_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.5)-1]
#q95_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.95)]
#q05_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.05)]

#plt.plot(x, q50, label='Dataset size projection', color=color_data)
#plt.fill_between(x, q05, q95, alpha=0.1, color=color_data)
#
#plt.plot(x, q50c, label='Dataset size projection (comp)', color=color_comp)
#plt.fill_between(x, q05c, q95c, alpha=0.1, color=color_comp)
plt.plot(x, q50a, label='Dataset size projection', color=colors[2])
plt.fill_between(x, q05a, q95a, alpha=0.1, color=colors[2])

#plt.axvline(median_runout_year, linestyle='dashed', color=color_data)
#plt.vlines(q95_runout_year, 1, 1e20, linestyle=':', color='blue', alpha=0.5)
#plt.vlines(q05_runout_year, 1, 1e20, linestyle=':', color='blue', alpha=0.5)

#plt.axvline(median_runout_year_comp, linestyle='dashed', color=color_comp)
#plt.vlines(q95_runout_year_comp, 1, 1e20, linestyle=':', color='red', alpha=0.5)
#plt.vlines(q05_runout_year_comp, 1, 1e20, linestyle=':', color='red', alpha=0.5)

plt.axvline(median_runout_year_agg, linestyle='dashed', color=colors[2], label="Median date of\nfull stock utilization")
#plt.axvline(q95_runout_year_agg, linestyle=':', color=color_comp, alpha=0.5)
#plt.axvline(q05_runout_year_agg, linestyle=':', color=color_comp, alpha=0.5)

model_dates = [
    2020.49,
    2022.26,
    2023.75,
    2021.66,
    2024.36,
    2024.27,
]

model_datas = [
    3.74e11,
    5.85e+11,
    2.625e12,
    1.87e+12,
    15e12,
    12e12,
]

models = [
    'GPT-3',
    'PaLM',
    'Falcon-180B',
    'FLAN 137B',
    'Llama 3',
    'DBRX',
]


plt.scatter(model_dates, model_datas, color=color_data, alpha=0.5)
plt.text(2020.4, 1.5e11, "GPT-3")
plt.text(2022.8, 5e11, "PaLM")
plt.text(2024.4, 2e12, "Falcon-180B")
plt.text(2020.4, 3e12, "FLAN")
plt.text(2022.1, 2.2e13, "Llama 3")
plt.text(2024.8, 0.8e13, "DBRX")

plt.legend()
plt.xticks(list(range(2020,2041,2)))
plt.yscale('log')
#plt.ylim(3e12, 1e16)
plt.xlim(2020, 2040)
plt.ylabel('Effective stock (number of tokens)')
plt.xlabel('Year')
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(ax.spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
#plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
#plt.gca().xaxis.set_major_locator(plt.NullLocator())
#plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('projections.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

fig = plt.figure(figsize=(5,5))
ax1, ax2 = fig.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

x = np.linspace(2020,2041,100)#list(range(2020, 2042))

ax1.plot(x, sq50, label='Stock of data', color='black', linestyle='dotted')
ax1.plot(x, sq95, linestyle=(0, (1,10)), color='black')
ax1.plot(x, sq05, linestyle=(0, (1,10)), color='black')

#run_out_probs = [s.mean() for s in samples_run_out]
#median_runout_year = x[np.searchsorted(run_out_probs, 0.5)]
#q95_runout_year = x[np.searchsorted(run_out_probs, 0.95)]
#q05_runout_year = x[np.searchsorted(run_out_probs, 0.05)]

#run_out_probs_comp = [s.mean() for s in samples_run_out_comp]
#median_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.5)]
#q95_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.95)]
#q05_runout_year_comp = x[np.searchsorted(run_out_probs_comp, 0.05)]

run_out_probs_agg = [s.mean() for s in samples_run_out_agg]
median_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.5)-1]
#q95_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.95)]
#q05_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.05)]

#plt.plot(x, q50, label='Dataset size projection', color=color_data)
#plt.fill_between(x, q05, q95, alpha=0.1, color=color_data)
#
#plt.plot(x, q50c, label='Dataset size projection (comp)', color=color_comp)
#plt.fill_between(x, q05c, q95c, alpha=0.1, color=color_comp)
ax1.plot(x, q50a, label='Dataset size projection', color=colors[2])
ax1.fill_between(x, q05a, q95a, alpha=0.1, color=colors[2])

#plt.axvline(median_runout_year, linestyle='dashed', color=color_data)
#plt.vlines(q95_runout_year, 1, 1e20, linestyle=':', color='blue', alpha=0.5)
#plt.vlines(q05_runout_year, 1, 1e20, linestyle=':', color='blue', alpha=0.5)

#plt.axvline(median_runout_year_comp, linestyle='dashed', color=color_comp)
#plt.vlines(q95_runout_year_comp, 1, 1e20, linestyle=':', color='red', alpha=0.5)
#plt.vlines(q05_runout_year_comp, 1, 1e20, linestyle=':', color='red', alpha=0.5)

ax1.axvline(median_runout_year_agg, linestyle='dashed', color=colors[2], label="Median date of\nfull stock utilization")
#plt.axvline(q95_runout_year_agg, linestyle=':', color=color_comp, alpha=0.5)
#plt.axvline(q05_runout_year_agg, linestyle=':', color=color_comp, alpha=0.5)

model_dates = [
    2020.49,
    2022.26,
    2023.75,
    2021.66,
    2024.36,
    2024.27,
]

model_datas = [
    3.74e11,
    5.85e+11,
    2.625e12,
    1.87e+12,
    15e12,
    12e12,
]

models = [
    'GPT-3',
    'PaLM',
    'Falcon-180B',
    'FLAN 137B',
    'Llama 3',
    'DBRX',
]


ax1.scatter(model_dates, model_datas, color=color_data, alpha=0.5)
ax1.text(2020.4, 1.5e11, "GPT-3")
ax1.text(2022.8, 5e11, "PaLM")
ax1.text(2024.4, 2e12, "Falcon-180B")
ax1.text(2020.4, 3e12, "FLAN")
ax1.text(2022.1, 2.2e13, "Llama 3")
ax1.text(2024.8, 0.8e13, "DBRX")

ax1.legend()
#ax1.set_xticks([list(range(2020,2041,2))])
ax1.set_yscale('log')
#ax1.set_ylim(3e12, 1e16)
#ax1.set_xlim(2020, 2040)
ax1.set_ylabel('Effective stock (number of tokens)')
#ax1.set_xlabel('Year')
ax1.grid('major', color='#F2F6F6', zorder=0)
plt.setp(ax1.spines.values(), color='#CCD8D9')
ax1.tick_params(axis='both', which='both', color='#CCD8D9')

tax2 = ax2.twinx()

pdf = np.diff(run_out_probs_agg, prepend=0)
window_width = 10

cumsum_vec = np.cumsum(np.insert(pdf, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

l1 = ax2.plot(x, run_out_probs_agg, label='Cumulative probability', color=color_data)
l2 = tax2.plot(x, np.concatenate(([0]*(window_width//2), ma_vec, [0]*(window_width//2-1))), label='Probability density', color=color_comp)
#plt.plot(x, run_out_probs_comp, label='Dataset size projection (comp)', color=color_comp)

lns = l1+l2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='center right')

ax2.set_xlim(2020,2040)
ax2.set_ylim(0,1.05)
tax2.set_ylim(0,ma_vec.max()*1.05)
ax2.set_ylabel('Probability of having exhausted\nstock of data')
#ax2.set_ylabel('Density')
ax2.set_xticks(list(range(2020,2041,2)))
tax2.set_yticks([])
ax2.set_xlabel('Year')
ax2.grid('major', color='#F2F6F6', zorder=0)
plt.setp(ax2.spines.values(), color='#CCD8D9')
plt.setp(tax2.spines.values(), color='#CCD8D9')
ax2.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0.0, wspace = 0)
fig.savefig('exhaustion_date.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()
