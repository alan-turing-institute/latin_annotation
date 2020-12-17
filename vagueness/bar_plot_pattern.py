import pandas as pd
import matplotlib.pyplot as plt

inputf = pd.read_csv('ratios_by_pattern.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())
pattern = columns[0]
alpha = columns[1]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Vagueness by derivation pattern')
ax.set_xlabel('Derivation pattern')
ax.set_ylabel('Vagueness score')

ax.bar(pattern,alpha, width=0.3)

plt.show()