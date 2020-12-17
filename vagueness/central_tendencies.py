import pandas as pd
import statistics


inputf = pd.read_csv('ratios_by_word.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())

alpha = columns[1]

#results = [abs(x) for x in alpha]
results = [x for x in alpha if x > 0] or None

print('mean: ', statistics.mean(results), '; median: ', statistics.median(results))
