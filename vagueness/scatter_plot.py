import pandas as pd
import matplotlib.pyplot as plt
import os

inputf = pd.read_csv('ratios_by_word.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())
words = columns[0]
alpha = columns[1]
beta = columns[2]


fig = plt.figure()
ax1 = fig.add_subplot(111)

#definition of both series
ax1.scatter(words, alpha, c="rebeccapurple")
#ax1.scatter(words, beta, c="red", label="beta")

#grid lines
ax1.xaxis.grid(True, linestyle=':')
ymajor = 0.0
yminor = 0.11
ax1.axhline(ymajor, color="black")
ax1.axhline(yminor, linestyle=':', color="black")

ax1.set_title('Vagueness by word')
ax1.set_xlabel('Word')
ax1.set_ylabel('Score')

#x axis
plt.xticks(rotation=60)

#legend
#plt.legend(loc='upper left')

outdir = 'charts'
fname = os.path.join(outdir, 'overview.png')
plt.savefig(fname, dpi=500, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.show()
