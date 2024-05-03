from utils import *

def sigmoid(x, l, x0, k, b):
    return (l / (1 + np.exp(-k*(x-x0))) + b)

# World population from 1950 to 2100, from OWID
human_pop_table = [2536605808,2584034223,2630861688,2677609059,2724846751,2773019915,2822443253,2873306055,2925686678,2979576143,3035160180,3091843506,3150420759,3211000941,3273978271,3339583509,3407922630,3478770102,3551599432,3625680964,3700685676,3775760030,3851650585,3927780518,4003794178,4079480473,4154666824,4229505917,4304533597,4380506180,4458274952,4536996616,4617386524,4699569184,4784011512,4870921665,4960567998,5052521994,5145425992,5237441433,5327529078,5414289382,5498919891,5581597596,5663150426,5744212929,5824891927,5905045647,5984794072,6064239030,6143776621,6222912459,6302062210,6381477292,6461454653,6542205330,6623819401,6706251239,6789396380,6873077808,6957137521,7041509491,7126144677,7210900157,7295610265,7380117870,7464344232,7548182589,7631091110,7713468203,7794798725,7874965730,7953952572,8031800338,8108605253,8184437453,8259276649,8333078315,8405863296,8477660720,8548487367,8618349452,8687227871,8755083512,8821862703,8887524228,8952048883,9015437614,9077693643,9138828561,9198847382,9257745483,9315508153,9372118246,9427555381,9481803269,9534854671,9586707748,9637357320,9686800144,9735033898,9782061757,9827885441,9872501561,9915905250,9958098746,9999085167,10038881262,10077518080,10115036358,10151469682,10186837206,10221149040,10254419002,10286658352,10317879312,10348098076,10377330829,10405590529,10432889134,10459239499,10484654857,10509150399,10532742861,10555450002,10577288193,10598274168,10618420909,10637736818,10656228231,10673904453,10690773333,10706852426,10722171373,10736765441,10750662352,10763874021,10776402016,10788248944,10799413362,10809892301,10819682642,10828780958,10837182075,10844878796,10851860143,10858111582,10863614774,10868347632,10872284132,10875393716]

def human_pop(t):
    return human_pop_table[t-1950]

# Fitted sigmoid to OWID data (https://ourworldindata.org/grapher/share-of-individuals-using-the-internet)
def internet_penetration(t):
    #return sigmoid(t, 0.7286, 2011.6, 0.16, 0)
    return sigmoid(t, 1, 2016.72, 0.1507, 0)

def internet_pop_normalized_2024(t):
    return human_pop(t) * internet_penetration(t) / (human_pop(2024) * internet_penetration(2024))

_cache_pop = dict()
def total_population_years(t):
    if t not in _cache_pop:
      _cache_pop[t] = sum([internet_pop_normalized_2024(y) for y in range(1950, int(floor(t)))]) + internet_pop_normalized_2024(int(ceil(t)))*(t-floor(t))
    return _cache_pop[t]

data = pd.read_csv('share-of-individuals-using-the-internet.csv')
years = data.year
pen = data.internet_pen

fig = plt.figure(figsize=(5,3))

plt.plot(list(range(1990,2015)), [human_pop(x) for x in list(range(1990,2015))], label='Human population', color=color_data)
plt.plot(list(range(2015,2050)), [human_pop(x) for x in list(range(2015,2050))], linestyle='--', color=color_data)

users_real_color = extended_colors[5]
users_fit_color = extended_colors[7]

internet_pop = [human_pop(y)/100*pen[years == y] for y in range(1990,2015)]
plt.plot(list(range(1990,2015)), internet_pop, label='Internet users (real)', color=users_real_color, zorder=10)

plt.plot(list(range(1990,2015)), [human_pop(x)*internet_penetration(x) for x in list(range(1990,2015))], label='Internet users (fit)', color=users_fit_color)
plt.plot(list(range(2015,2050)), [human_pop(x)*internet_penetration(x) for x in list(range(2015,2050))], linestyle='--', color=users_fit_color)

plt.legend()
plt.xlabel('Year')
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(plt.gca().spines.values(), color='#CCD8D9')
plt.yticks([2e9, 4e9, 6e9, 8e9], ['2B', '4B', '6B', '8B'])
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.margins(0,0)
plt.gcf().savefig('internet_users.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

def popular_platforms_2024():

    # source: https://blog.gdeltproject.org/visualizing-twitters-evolution-2012-2020-and-how-tweeting-is-changing-in-the-covid-19-era/
    tw_users = 3e8
    tw_posts_per_userday = 1.3

    # Median is around 7 according to https://blog.gdeltproject.org/visualizing-twitters-evolution-2012-2020-and-how-tweeting-is-changing-in-the-covid-19-era/
    tw_avg_length = sq.to(2,25, credibility=95)

    tw_tokens_2024 = tw_users * tw_posts_per_userday * 365 * tw_avg_length
    tw_traffic_share = 0.013

    # source: https://datareportal.com/essential-facebook-stats
    fb_users = 3e9
    fb_posts_per_usermonth = 1
    fb_comments_per_usermonth = 5
    fb_avg_post_length = sq.to(40, 90) # Median 60, from https://github.com/jerryspan/FacebookR/
    fb_avg_comment_length = sq.to(17, 40) # Median 26, from https://github.com/jerryspan/FacebookR/

    fb_tokens_2024 = fb_users * (
            fb_posts_per_usermonth*fb_avg_post_length
            + fb_comments_per_usermonth*fb_avg_comment_length)
    fb_traffic_share = 0.036

    # Source: Data Never Sleeps (Domo)
    ig_posts_per_min = 66e3
    ig_avg_post_length = sq.to(10,40)

    ig_tokens_2024 = ig_posts_per_min * 60 * 24 * 365 * ig_avg_post_length
    ig_traffic_share = 0.015

    # Source: https://arxiv.org/pdf/2001.08435.pdf
    reddit_posts_per_day = 6e5
    reddit_comms_per_day = 5e6
    reddit_avg_post_length = sq.to(32, 72)
    reddit_avg_comm_length = sq.to(14, 31)

    reddit_tokens_2024 = 365*(reddit_posts_per_day*reddit_avg_post_length + reddit_comms_per_day*reddit_avg_comm_length)
    reddit_traffic_share = 0.0044

    #total_tokens_2024 = (
    #        tw_tokens_2024 / tw_traffic_share
    #        *fb_tokens_2024 / fb_traffic_share
    #        *ig_tokens_2024 / ig_traffic_share
    #        *reddit_tokens_2024 / reddit_traffic_share
    #        )**(1/4)
    total_tokens_2024 = sq.mixture([
            tw_tokens_2024 / tw_traffic_share,# * sq.to(1/3, 3),
            fb_tokens_2024 / fb_traffic_share,# * sq.to(1/3, 3),
            ig_tokens_2024 / ig_traffic_share,# * sq.to(1/3, 3),
            reddit_tokens_2024 / reddit_traffic_share,# * sq.to(1/3, 3),
    ])#* sq.to(1/10, 10)

    return total_tokens_2024

def instant_messages_2024():
    return sq.to(1e11, 2e11) * 365 * sq.to(4,6)

def stock_internet_users(t):
    return total_population_years(t)*(popular_platforms_2024() + instant_messages_2024())

plt.hist(np.log10(popular_platforms_2024() @ 100_000), bins=100, alpha=0.4)
plt.hist(np.log10(instant_messages_2024() @ 100_000), bins=100, alpha=0.4)
plt.hist(np.log10((popular_platforms_2024() + instant_messages_2024()) @ 100_000), bins=100, alpha=0.4)

print(np.quantile((popular_platforms_2024() + instant_messages_2024()) @ 100_000, 0.5) / 1e12)
print(np.quantile((popular_platforms_2024() + instant_messages_2024()) @ 100_000, 0.025) / 1e12)
print(np.quantile((popular_platforms_2024() + instant_messages_2024()) @ 100_000, 0.975) / 1e12)
plt.show()

fig = plt.figure()
ax = fig.subplots(1)

x = np.linspace(2024, 2040, 17)

cc = [stock_indexed_web(y) @ 10_000 for y in x]
sq95cc = [np.quantile(s, 0.95) for s in cc]
sq50cc = [np.quantile(s, 0.50) for s in cc]
sq05cc = [np.quantile(s, 0.05) for s in cc]

iu = [stock_internet_users(y) @ 10_000 for y in x]
sq95iu = [np.quantile(s, 0.95) for s in iu]
sq50iu = [np.quantile(s, 0.50) for s in iu]
sq05iu = [np.quantile(s, 0.05) for s in iu]

plt.plot(x, sq50cc, label='CC', color = colors[0])
plt.fill_between(x, sq05cc, sq95cc, alpha=0.3, color=colors[0])

plt.plot(x, sq50iu, label='IU', color = colors[3])
plt.fill_between(x, sq05iu, sq95iu, alpha=0.3, color=colors[3])

plt.legend(loc="upper left")
plt.yscale('log')
plt.xlim(2024, 2040)
plt.ylabel('Effective stock (number of words)')
plt.xlabel('Year')
plt.grid('major', color='#F2F6F6', zorder=0)
plt.setp(ax.spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.show()

plt.hist(np.log10(stock_indexed_web(2024) @ 100_000), bins=50, alpha=0.5)
plt.hist(np.log10(stock_internet_users(2024) @ 100_000), bins=50, alpha=0.5)
plt.show()






import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

fig, [ax1, ax2] = plt.subplots(2,1, figsize=(5,7))

def expmodel(x, A, x0, k, b):
    return A*np.exp(k*(x-x0)) + b

def sigmoid2(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))

def sigmexp(x, A, x0e, ke, be, x0s, ks,):
    return expmodel(x, A, x0e, ke, be) * sigmoid2(x, x0s, ks)

def linear(x, m, n):
    return x*m + n

dt = pd.read_csv('reddit_open_web_text.csv')
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
fig.savefig('reddit.pdf')
plt.show()

