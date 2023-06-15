import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file = 'results/33bus_4MGs/33bus_4MGs_gub0.010_2022_05_18_20_37_19_cg.csv'

figsize = (16, 6)

data = pd.read_csv(file)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

avg_span = int(288 * 5)
c = data['c']
g = data[' g']
data_length = len(c) // avg_span
c_avg = [np.mean(c[i * avg_span: (i + 1) * avg_span]) for i in range(data_length)]
g_avg = [np.mean(g[i * avg_span: (i + 1) * avg_span]) for i in range(data_length)]


plt.figure(1, figsize=figsize)
plt.plot(c_avg)
plt.legend(prop=font)
plt.show()

plt.figure(2, figsize=figsize)
plt.ylim(0, 1)
plt.plot(g_avg)
plt.legend(prop=font)
plt.show()
