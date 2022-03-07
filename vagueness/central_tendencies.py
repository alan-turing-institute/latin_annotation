import pandas as pd
import statistics as st
import numpy as np
from numpy import mean, absolute

from statsmodels import robust


inputf = pd.read_csv('ratios_by_word.tsv', sep="\t", header=None)
columns = []
for column in inputf:
    columns.append(inputf[column].tolist())

alpha = columns[1]

alpha = sorted(alpha)

results = [abs(x) for x in alpha]
results_natural = [x for x in alpha if x > 0] or None




# First quartile (Q1)
Q1 = np.median(alpha[:20])


# Third quartile (Q3)
Q3 = np.median(alpha[20:])
  
# Interquartile range (IQR)
IQR = Q3 - Q1

sample = alpha[6:34]

# First quartile (Q1)
Q1s = np.median(sample[:14])

# Third quartile (Q3)
Q3s = np.median(sample[14:])

# Interquartile range (IQR)
IQRs = Q3s - Q1s

#median absolute deviation
MAD = robust.mad(alpha, c=1)

MADs = robust.mad(sample, c=1)

#absolute mean deviation
AMD = mean(absolute(alpha - mean(alpha)))

AMDs = mean(absolute(sample - mean(sample)))

high = np.median(alpha) + MAD

low = np.median(alpha) - MAD

print("Standard Deviation: ", st.stdev(alpha))

print("Standard Deviation (sample): ", st.stdev(sample))

print("Variance (total): ", st.pvariance(alpha))

print("Variance (sample): ", st.variance(sample))
  
print('IQR: ', MAD)

print('mean: ', np.mean(alpha), '; median: ', np.median(alpha))
print('mean (sample): ', np.mean(sample), '; median (sample): ', np.median(sample))


print('High: ', high, '/ Low: ', low)
