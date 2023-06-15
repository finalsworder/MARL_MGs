from env import distribution_network
from env.DN_4bus_2MGs import DN_4bus_2MGs
from env.DN_2bus_1MG import DN_2bus_1MGs
from rl_torch.cmdp_onpolicy_v2 import multi_agent_system
import torch
from rl_torch.nn import Actor, Critic, Actor_state, Critic_state
import numpy as np
from copy import copy
import random
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsiode_duration = int(288 * 10)
batch_size = 256
iterations = 10
exploration = .0
max_episode = int(1e5)
stepsize_mu_g = 1e-4
g_ub = [0.1]

env = DN_2bus_1MGs()
MGs_action_space = []
MGs_action_min = []
MGs_action_max = []
for MG in env.MGs:
    MGs_action_space.append(MG.action_dim_generator + MG.action_dim_storage)
    action_min, action_max = MG.get_action_range()
    MGs_action_min.append(action_min)
    MGs_action_max.append(action_max)

action_all = [np.zeros(MGs_action_space[i]) for i in range(env.MG_num)]
state, sequence = env.step(action_all)
state_dim = len(state)
sequence_dim = sequence.shape[0]
env.calculate_cost()
c_episode = []
g_episode = []
c = env.c
g = env.g
# g = np.array(g).reshape(env.MG_num, 1)
# w = [[.9, .1], [.1, .9]]
w = [[1.]]
agents = multi_agent_system(agent_num=env.MG_num, device=device, communication_matrix=w,
                            actor_fn=Actor_state, critic_fn=Critic_state, state_dim=state_dim,
                            sequence_dim=sequence_dim, action_dims=MGs_action_space, lr_c=1e-3, lr_a=1e-3,
                            constraint_num=1, gamma=.99, tau=.99,
                            memory_volume=100000,
                            action_min=MGs_action_min, action_max=MGs_action_max, g_ub=g_ub)
mu_g = np.zeros((agents.agent_num, agents.constraint_num))
dual_variable = np.zeros((agents.agent_num, agents.constraint_num))

time_step = 0
for epsiode in range(max_episode):
    c_t = np.zeros(epsiode_duration)
    g_t = np.zeros(epsiode_duration)
    for t in range(epsiode_duration):
        if random.uniform(0, 1) > exploration:
            action_next_all = agents.step(state, sequence)
        else:
            # action_next_all = [[1, 0.5, 0, 0]]
            action_next_all = np.random.uniform(0, 1, (1, 4))
        state_, sequence_ = env.step(action_next_all)
        agents.store(state, sequence, action_all, c, g, state_, sequence_, action_next_all)
        state = copy(state_)
        sequence = copy(sequence_)
        action_all = copy(action_next_all)
        env.calculate_cost()
        c = env.c
        g = env.g
        c_t[t] = np.array(c).sum()
        g_t[t] = np.array(g).sum()
        # stepsize_dual = 1e5 / (1e5 + time_step)
        # for i in range(agents.constraint_num):
        #     for j, agent in enumerate(agents.agents):
        #         mu_g[j][i] = (1 - stepsize_mu_g) * mu_g[j][i] + stepsize_mu_g * g[j][i]
        # mu_g_copy = copy(mu_g)
        # for i in range(agents.constraint_num):
        #     for j, agent in enumerate(agents.agents):
        #         mu_g[j][i] = sum([mu_g_copy[k][i] * w[j][k] for k in range(agents.agent_num)])
    c_episode.append(c_t.mean())
    g_episode.append(g_t.mean())
    for _ in range(iterations):
        # for i in range(agents.constraint_num):
        #     for j, agent in enumerate(agents.agents):
        #         dual_variable[j][i] = max(0, dual_variable[j][i] + stepsize_mu_g * (mu_g[j][i] - g_ub[i]))
        #
        # dual_variable_copy = copy(dual_variable)
        # for i in range(agents.constraint_num):
        #     for j, agent in enumerate(agents.agents):
        #         dual_variable[j][i] = sum([dual_variable_copy[k][i] * w[j][k] for k in range(agents.agent_num)])
        #         agent.dual_variables[i] = dual_variable[j][i]
        agents.train_batch(batch_size=batch_size, critic_iter=1, actor_iter=1)
    print('episode %d: c_avg: %.3f, g_avg: %.3f' % (epsiode, c_episode[-1], g_episode[-1]))


