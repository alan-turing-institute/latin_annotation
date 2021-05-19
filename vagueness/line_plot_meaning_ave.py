import pandas as pd
import matplotlib.pyplot as plt

inputf = pd.read_csv('ratios_by_meaning_nbr.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())
source = columns[0]
alpha = columns[1]
source = map(str, source)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Vagueness by number of meanings (average)')
ax.set_xlabel('Number of meanings')
ax.set_ylabel('Vagueness score')

ax.plot(list(source), alpha, linestyle='solid')

plt.show()