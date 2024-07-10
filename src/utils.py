import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import squigglepy as sq
from datetime import datetime
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams['figure.dpi'] = 150
plt.rcParams["figure.figsize"] = (4.5,3.5)

import matplotlib as mpl
#mpl.use('PDF')
# Ensure Type 1 fonts are used
plt.rcParams['ps.useafm'] = True
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

# You might have to clear Matplotlib's font cache for the custom font to be found
#plt.rcParams['font.family'] = 'Messina sans'

colors = [
  '#034752',
  '#02767C',
  '#00A5A6',
  '#11DF8C',
  '#93E75E',
  '#C5F1AB',
  '#DCE67A',
  '#FFDE5C',
]

extended_colors = [
  '#47BED6',
  '#1F95BD',
  '#086F91',
  '#034752',
  '#02767C',
  '#00A5A6',
  '#00AF77',
  '#11DF8C',
  '#93E75E',
  '#B9EE98',
  '#FFDE5C',
  '#FFB45C',
  '#FF975C',
]


model_fill_color = '#9BBFC1'
model_fill_alpha = 0.8

data_stock_color = '#00A5A6'
data_stock_ci_alpha = 0.15

compute_projection_color = data_stock_color
compute_projection_ci_alpha = data_stock_ci_alpha

dataset_size_color = '#1F6EE5'
dataset_size_ci_alpha = 0.15

historical_projection_color = dataset_size_color
historical_projection_ci_alpha = dataset_size_ci_alpha

primary_usage_date_color = '#E03D90'
secondary_usage_date_color = '#6A3ECB'

human_pop_color = '#00A5A6'
internet_users_real_color = '#FFB23C'
internet_users_fit_color = '#1F6EE5'

title_props = {
    #'fontname': 'serif',
    'pad': 8,
}

top_x_label_props = {
    'labelpad': 11,
}

color_data = colors[0]
color_comp = colors[3]


def hex_to_rgba(hex, alpha=1.0):
    hex = hex.lstrip('#')
    h_len = len(hex)
    return tuple(int(hex[i:i + h_len // 3], 16) / 255 for i in range(0, h_len, h_len // 3)) + (alpha,)
