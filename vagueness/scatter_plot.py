import pandas as pd
import matplotlib.pyplot as plt

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
ax1.scatter(words, alpha, c="blue", label="alpha")
ax1.scatter(words, beta, c="red", label="beta")

#grid lines
ax1.xaxis.grid(True, linestyle=':')
ymajor = 0.0
yminor = 0.2
ax1.axhline(ymajor, color="black")
ax1.axhline(yminor, linestyle=':', color="black")

ax1.set_title('Sense identification by word')
ax1.set_xlabel('Word')
ax1.set_ylabel('Score')

#x axis
plt.xticks(rotation=60)

#legend
plt.legend(loc='upper left')

plt.show()
