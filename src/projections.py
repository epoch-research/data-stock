"""# Full projections"""

from stock_adjust import *
from dataset_size_projections import get_dataset_size_projections
from profit_maximization import get_optimal_policies, numerical_optscal_data

dataset_size, dataset_size_hist, dataset_size_comp = get_dataset_size_projections()

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

def dataset_size_overtraining(y, ns, ds):
    opt_data = dataset_size(y)
    comp = np.log10(6/20*(opt_data**2))
    return 10**numerical_optscal_data(comp, np.log10(6*ns*ds), np.log10(ds))

def aggregate_projection_overtraining(y, dataset_size):
    stock = aggregate(y) @ 10_000
    stock_quantiles = [np.quantile(stock, q) for q in np.linspace(0,1,101)]
    dataset = []
    run_out = []
    for i in range(100):
        samples = dataset_size((y-2024)*365).clip(0, stock_quantiles[i+1])
        dataset.append(samples)
        run_out.append(np.isclose(samples, stock_quantiles[i+1]))
    return np.concatenate(dataset), np.concatenate(run_out)

def smooth(data):
  window_size = 5
  # Create a convolution kernel for the moving average
  kernel = np.ones(window_size) / window_size
  # Apply the convolution to the data
  smoothed_data = np.convolve(data, kernel, mode='valid')

  # Pad the smoothed data to match the original data length
  pad_size = (len(data) - len(smoothed_data)) // 2
  smoothed_data = np.pad(smoothed_data, (pad_size, len(data) - len(smoothed_data) - pad_size), mode='edge')
  return smoothed_data

def plot_projection_double(x, sq50, sq95, sq05, proj1, proj2, fname="projections.pdf", dt_min_year=2020, figsize=(5,3)):

  fig = plt.figure(figsize=figsize)
  ax = fig.subplots(1)

  samples_run_out_agg, q50a, q95a, q05a = proj1

  run_out_probs_agg = [s.mean() for s in samples_run_out_agg]
  median_runout_year_agg = x[np.searchsorted(run_out_probs_agg, 0.5)-1]

  window_size = 3

  plt.plot(x, smooth(sq50), label='Stock of data', color=data_stock_color, alpha=0.8, zorder=200)
  plt.fill_between(x, smooth(sq05), smooth(sq95), alpha=data_stock_ci_alpha, color=data_stock_color, zorder=200)
  plt.plot(x, smooth(sq95), linestyle=(0, (2,1)), color=data_stock_color, alpha=0.8, zorder=200)
  plt.plot(x, smooth(sq05), linestyle=(0, (2,1)), color=data_stock_color, alpha=0.8, zorder=200)

  idx = x > dt_min_year
  plt.plot(x[idx], smooth(np.array(q50a)[idx]), label='Dataset size projection', color=dataset_size_color, alpha=0.8, zorder=200)
  plt.fill_between(x[idx], smooth(np.array(q05a)[idx]), smooth(np.array(q95a)[idx]), alpha=dataset_size_ci_alpha, color=dataset_size_color, zorder=200)
  plt.plot(x[idx], smooth(np.array(q05a)[idx]), linestyle=(0, (2,1)), color=dataset_size_color, alpha=0.8, zorder=200)
  plt.plot(x[idx], smooth(np.array(q95a)[idx]), linestyle=(0, (2,1)), color=dataset_size_color, alpha=0.8, zorder=200)

  samples_run_out_agg, q50a, q95a, q05a = proj2

  run_out_probs_agg_over = [s.mean() for s in samples_run_out_agg]
  median_runout_year_agg_over = x[np.searchsorted(run_out_probs_agg_over, 0.5)-1]


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


  plt.scatter(model_dates, model_datas, color=model_fill_color, alpha=model_fill_alpha)
  plt.text(2020.3, 2.4e11, "GPT-3")
  plt.text(2022.45, 5.5e11, "PaLM")
  plt.text(2023.9, 2.4e12, "Falcon-180B")
  plt.text(2021.9, 1.7e12, "FLAN")
  plt.text(2024.2, 1.9e13, "Llama 3")
  plt.text(2024.3, 8e12, "DBRX")

  #plt.legend()
  plt.xticks(list(range(2020,2035,1)))
  plt.yscale('log')
  plt.yticks([], [], minor=True)
  plt.ylim(8e10, 5e15)
  plt.xlim(2020, 2034)

  x0, x1 = plt.xlim()
  y0, y1 = plt.ylim()

  # Create a custom colormap from #E03D90 to transparent
  colors = [hex_to_rgba('#E03D90', alpha) for alpha in np.linspace(1, 0, 256)]
  cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)

  xs = np.array([median_runout_year_agg, median_runout_year_agg+0.02])
  ys = np.logspace(np.log10(y0), np.log10(y1), 50)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))
  plt.pcolormesh(xs, ys, zs, shading='gouraud', cmap=cmap, vmin=0, vmax=1, zorder=300)

  pct_10_year = x[np.searchsorted(run_out_probs_agg, 0.1)-1]
  pct_90_year = x[np.searchsorted(run_out_probs_agg, 0.9)-1]
  X,Y = np.meshgrid(np.linspace(pct_10_year, pct_90_year, 200), np.logspace(np.log10(y0), np.log10(y1), 200))
  z = (np.log10(Y)- np.log10(y0)+3)/(np.log10(y1)-np.log10(y0))
  plt.pcolormesh(X, Y, z, cmap=cmap, vmin=0, vmax=1, shading='auto', zorder=10)

  colors = [hex_to_rgba('#E03D90', alpha) for alpha in np.concatenate((np.linspace(1, 0, 100)*np.array([1,0]*50), np.array([0]*50)))]
  cmap = mpl.colors.ListedColormap(colors)

  xs = np.array([pct_10_year-0.02, pct_10_year])
  ys = np.logspace(np.log10(y0), np.log10(y1), 150)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)+1) / (np.log10(y1) - np.log10(y0)) + 1/149
  plt.pcolormesh(xs, ys, zs[:-1,:-1], shading='flat', cmap=cmap, vmin=0, vmax=1, zorder=300)

  xs = np.array([pct_90_year+0.02, pct_90_year+0.04])
  ys = np.logspace(np.log10(y0), np.log10(y1), 150)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)+1) / (np.log10(y1) - np.log10(y0)) + 1/149
  plt.pcolormesh(xs, ys, zs[:-1,:-1], shading='flat', cmap=cmap, vmin=0, vmax=1, zorder=300)


  # Create a custom colormap from #6A3ECB to transparent
  colors = [hex_to_rgba('#6A3ECB', alpha) for alpha in np.linspace(1, 0, 256)]
  cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)

  xs = np.array([median_runout_year_agg_over, median_runout_year_agg_over+0.02])
  ys = np.logspace(np.log10(y0), np.log10(y1), 50)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)+2) / (np.log10(y1) - np.log10(y0))
  plt.pcolormesh(xs, ys, zs, shading='gouraud', cmap=cmap, vmin=0, vmax=1, zorder=300)

  pct_10_year = x[np.searchsorted(run_out_probs_agg_over, 0.1)-1]
  pct_90_year = x[np.searchsorted(run_out_probs_agg_over, 0.9)-1]
  X,Y = np.meshgrid(np.linspace(pct_10_year, pct_90_year, 200), np.logspace(np.log10(y0), np.log10(y1), 200))
  z = (np.log10(Y)- np.log10(y0)+4)/(np.log10(y1)-np.log10(y0))
  plt.pcolormesh(X, Y, z, cmap=cmap, vmin=0, vmax=1, shading='auto', zorder=10)

  colors = [hex_to_rgba('#6A3ECB', alpha) for alpha in np.concatenate((np.linspace(1, 0, 70)*np.array([1,0]*35), np.array([0]*80)))]
  cmap = mpl.colors.ListedColormap(colors)

  xs = np.array([pct_10_year-0.02, pct_10_year])
  ys = np.logspace(np.log10(y0), np.log10(y1), 150)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)+1) / (np.log10(y1) - np.log10(y0)) + 1/149
  plt.pcolormesh(xs, ys, zs[:-1,:-1], shading='flat', cmap=cmap, vmin=0, vmax=1, zorder=300)

  xs = np.array([pct_90_year+0.01, pct_90_year+0.03])
  ys = np.logspace(np.log10(y0), np.log10(y1), 150)
  xs, ys = np.meshgrid(xs, ys)
  zs = (np.log10(ys) - np.log10(y0)+1) / (np.log10(y1) - np.log10(y0)) + 1/149
  plt.pcolormesh(xs, ys, zs[:-1,:-1], shading='flat', cmap=cmap, vmin=0, vmax=1, zorder=300)

  plt.text(2022.8, 2e15, "Estimated stock of human-\ngenerated public text; 95\% CI", fontsize=11)
  plt.text(2020.2, 2.2e13, "Dataset sizes used to train\nnotable LLMs; 95\% CI", fontsize=11)
  plt.text(2026.45, 8e11, "$\sim$2027", color='#6A3ECB', fontsize=13, fontweight="heavy")
  plt.text(2024.75, 3.3e11, "Median date with 5x\novertraining; 80\% CI", fontsize=11)
  plt.text(2028.9, 6e12, "$\sim$2028", color='#E03D90', fontsize=13, fontweight="heavy")
  plt.text(2028.9, 2.6e12, "Median date of full\nstock use; 80\% CI", fontsize=11)

  style = "Simple, tail_width=0.1, head_width=3, head_length=4"
  kw = dict(arrowstyle=style, color="k", zorder=600)
  a1 = mpl.patches.FancyArrowPatch((2022.3, 1.6e13), (2023.22, 1.3e13),
                             connectionstyle="arc3,rad=.2", **kw)
  a2 = mpl.patches.FancyArrowPatch((2023.3, 1.7e15), (2023.7, 5e14),
                             connectionstyle="arc3,rad=.1", **kw)

  for a in [a1, a2]:
    plt.gca().add_patch(a)

  plt.ylabel('Effective stock (number of tokens)')#, ha='left', y=1.01, rotation=0, labelpad=0, fontsize=13)
  plt.xlabel('Year', fontsize=13)
  plt.grid('major', color='#F2F6F6', zorder=0)
  plt.setp(ax.spines.values(), color='#CCD8D9')
  plt.tick_params(axis='both', which='both', color='#CCD8D9')
  plt.tick_params(axis="y",direction="in")
  plt.tight_layout()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.savefig('results/'+fname, bbox_inches = 'tight', pad_inches=0)
  plt.show()


if __name__ == '__main__':
    C, ns1, ds1, ps1, ns2, ds2, ps2, ns3, ds3, ps3 = get_optimal_policies()

    x = np.linspace(2015,2039,100)#list(range(2020, 2042))

    stock = [aggregate(y) @ 10_000 for y in x]
    sq95 = [np.quantile(s, 0.95) for s in stock]
    sq50 = [np.quantile(s, 0.50) for s in stock]
    sq05 = [np.quantile(s, 0.05) for s in stock]

    samples_agg, samples_run_out_agg = zip(*[aggregate_projection(y) for y in x])
    q95a = [np.quantile(s, 0.95) for s in samples_agg]
    q50a = [np.quantile(s, 0.50) for s in samples_agg]
    q05a = [np.quantile(s, 0.05) for s in samples_agg]

    samples_agg_over, samples_run_out_agg_over1 = zip(*[aggregate_projection_overtraining(y, lambda y: dataset_size_overtraining(y, ns1, ds1)) for y in x])
    q50ao1 = np.array([np.quantile(s, 0.50) for s in samples_agg_over])
    samples_agg_over, samples_run_out_agg_over2 = zip(*[aggregate_projection_overtraining(y, lambda y: dataset_size_overtraining(y, ns2, ds2)) for y in x])
    q50ao2 = np.array([np.quantile(s, 0.50) for s in samples_agg_over])
    q05ao2 = np.array([np.quantile(s, 0.05) for s in samples_agg_over])
    q95ao2 = np.array([np.quantile(s, 0.95) for s in samples_agg_over])
    samples_agg_over, samples_run_out_agg_over3 = zip(*[aggregate_projection_overtraining(y, lambda y: dataset_size_overtraining(y, ns3, ds3)) for y in x])
    q50ao3 = np.array([np.quantile(s, 0.50) for s in samples_agg_over])

    run_out_probs_agg1 = [s.mean() for s in samples_run_out_agg_over1]
    median_runout_year_agg1 = x[np.searchsorted(run_out_probs_agg1, 0.5)-1]
    print(median_runout_year_agg1)

    run_out_probs_agg2 = [s.mean() for s in samples_run_out_agg_over2]
    median_runout_year_agg2 = x[np.searchsorted(run_out_probs_agg2, 0.5)-1]
    print(median_runout_year_agg2)

    run_out_probs_agg3 = [s.mean() for s in samples_run_out_agg_over3]
    median_runout_year_agg3 = x[np.searchsorted(run_out_probs_agg3, 0.5)-1]
    print(median_runout_year_agg3)

    plot_projection_double(x, sq50, sq95, sq05, (samples_run_out_agg, q50a, q95a, q05a), (samples_run_out_agg_over2, q50ao2, q95ao2, q05ao2), fname="projections.pdf", figsize=(5*1.4,3*1.4), dt_min_year=2015)
