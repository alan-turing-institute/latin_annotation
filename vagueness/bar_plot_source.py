import pandas as pd
import matplotlib.pyplot as plt

inputf = pd.read_csv('ratios_by_source_nbr.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())
source = columns[0]
alpha = columns[1]
source = map(str, source)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Vagueness by number of source meanings')
ax.set_xlabel('Number of sources')
ax.set_ylabel('Vagueness score')



ax.bar(list(source),alpha, width=0.4)

plt.show()