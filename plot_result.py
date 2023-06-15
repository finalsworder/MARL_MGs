import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# file = 'results/4bus_2MGs/4bus_2MGs2022_05_06_20_23_58 (1).csv'
# file = 'results/33bus_4MGs/33bus_4MGs_gub2.00_2022_05_08_16_35_16.csv'
# file = 'results/4bus_2MGs/4bus_2MGs_gub10.00_2022_05_09_15_30_24.csv'
# file = 'results/2bus_1MG/2bus_1MG2022_05_12_18_31_22.csv'
# file = 'results/2bus_1MG/2bus_1MG2022_05_13_15_49_01.csv'
file = 'results/2bus_1MG/2bus_1MG2022_05_13_13_19_47.csv'
# file = 'results/33bus_4MGs/33bus_4MGs_state_gub2.00_2022_05_09_15_21_37.csv'

figsize = (16, 6)

data = pd.read_csv(file)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

start_t = int(2500 * 288)
end_t = start_t + 288 * 3
etn_price = data['etn_price']
plt.figure(1, figsize=figsize)
plt.plot(etn_price[start_t:end_t])
plt.legend(prop=font)
plt.show()

p_DG_data = data['p_MG_DG_0_0']
q_DG_data = data['q_MG_DG_0_0']
plt.figure(2, figsize=figsize)
plt.plot(p_DG_data[start_t:end_t], label='p_DG')
plt.plot(q_DG_data[start_t:end_t], label='q_DG')
plt.legend(prop=font)
plt.show()

p_ES_data = data['p_MG_ES_0_0']
q_ES_data = data['q_MG_ES_0_0']
S_DG_data = data['S_MG_ES_0_0']
plt.figure(3, figsize=figsize)
plt.plot(p_ES_data[start_t:end_t], label='p_ES')
plt.plot(q_ES_data[start_t:end_t], label='q_ES')
plt.plot(S_DG_data[start_t:end_t], label='S_ES')
plt.legend(prop=font)
plt.show()

# q_inj_data = data['q_inj_1']
# q_MG_DG = data['q_MG_DG_0_0']
# q_MG_ES = data['q_MG_ES_0_0']
# q_MG_load = data['q_MG_load_0_0']
# plt.figure(4, figsize=figsize)
# plt.plot(q_inj_data[start_t:end_t], label='q_inj_data')
# plt.plot(q_MG_DG[start_t:end_t], label='q_MG_DG')
# plt.plot(q_MG_ES[start_t:end_t], label='q_MG_ES')
# plt.plot(q_MG_load[start_t:end_t], label='q_MG_load')
# plt.legend(prop=font)
# plt.show()

agent_num = 1
c = [data['c_%d' % i] for i in range(agent_num)]
c_avg = []
episode_len = 288
plot_episode = len(c[0]) // episode_len
for ep in range(plot_episode):
    c_avg.append(np.sum([c[i][ep * episode_len: (ep + 1) * episode_len].mean() for i in range(agent_num)]))
plt.figure(5)
plt.plot(c_avg)
plt.show()



