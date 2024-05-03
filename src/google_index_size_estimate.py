from utils import *

df = pd.read_csv('google_pivot_words.csv')
for c in df.columns[1:-1]:
    df[c] = df['count']/(df[c]*0.45)

fig = plt.figure()
ax = fig.subplots(1)
plt.hist(df.iloc[:, 1:-1].to_numpy().flatten() / 1e9, bins=50, color=color_data)
plt.xlabel('Index size estimate (billions of web pages)')
plt.ylabel('Count')
plt.setp(ax.spines.values(), color='#CCD8D9')
plt.tick_params(axis='both', which='both', color='#CCD8D9')
plt.tight_layout()
plt.savefig('histogram.pdf', bbox_inches = 'tight', pad_inches=0)
plt.show()

