import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputf = pd.read_csv('competition_by_word_chron.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())
words = columns[0]
bc = columns[1]
ad = columns[2]


bar_width = 0.25
pos1 = np.arange(len(bc))
pos2 = [x + bar_width for x in pos1]


fig = plt.figure()
ax = fig.add_subplot(111)

#definition of both series
plt.bar(pos1, bc, color="rebeccapurple", label="BC", width = bar_width)
plt.bar(pos2, ad, color="plum", label="AD", width = bar_width)


#grid lines
ax.xaxis.grid(True, linestyle=':')
ymajor = 0.0
ax.axhline(ymajor, color="black")

ax.set_title('Vagueness by word (BC / AD)')
ax.set_xlabel('Word')
ax.set_ylabel('Score')

#x axis
plt.xticks([pos + bar_width for pos in range(len(pos1))], words,rotation=60)

#legend
plt.legend(loc='upper left')

plt.show()
