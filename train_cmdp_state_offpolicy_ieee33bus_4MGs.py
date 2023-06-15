from env import distribution_network
from env.IEEE_33bus_4MGs import DN_33bus_4MGs
from rl_torch.cmdp_offpolicy import multi_agent_system
import torch
from rl_torch.nn import Actor_sequence, Critic_sequence, Actor_state, Critic_state,\
    Actor_sequence_large, Critic_sequence_large
import numpy as np
from copy import copy
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import os

if __name__ == '__main__':
    EPISODES = int(2e4)
    max_episode_length = 288
    device = torch.device(f"cuda:{input('gpu:')}" if torch.cuda.is_available() else "cpu")
    epsiode_duration = int(288 * 3)
    batch_size = 1024
    buffer_size = int(1e6)
    start_train = 10 * 288
    train_interval = 100
    action_var = .3
    action_var_decay = .995
    action_var_min = .2
    g_ub = 2
    # record_min = 1e4
    record_interval = 10
    logfile = open('results/33bus_4MGs/33bus_4MGs_state_gub%2.2f_' % g_ub
                   + time.strftime('%Y_%m_%d_%H_%M_%S.csv'), 'w+', newline='')
    env = DN_33bus_4MGs()

    MGs_action_space = []
    MGs_action_min = []
    MGs_action_max = []
    for MG in env.MGs:
        MGs_action_space.append(MG.action_dim_generator + MG.action_dim_storage)
        action_min, action_max = MG.get_action_range()
        MGs_action_min.append(action_min)
        MGs_action_max.append(action_max)

    action_all = [np.zeros(MGs_action_space[i]) for i in range(env.MG_num)]
    state_, sequence_, _, _ = env.step(action_all)
    state_dim = len(state_)
    sequence_dim = sequence_.shape[0]
    env.calculate_cost()
    env.create_log_file(log_file=logfile)
    c = env.c
    g = env.g

    w = [[.95, .05, 0, 0],
         [0, .95, .05, 0],
         [0, 0, .95, .05],
         [.05, 0, 0, .95]]

    writer = SummaryWriter(log_dir='tensorboard_log/4bus_2MGs/' + time.strftime('%Y_%m_%d_%H_%M_%S'))
    agents = multi_agent_system(agent_num=env.MG_num, device=device, communication_matrix=w,
                                actor_fn=Actor_state, critic_fn=Critic_state, state_dim=state_dim,
                                sequence_dim=sequence_dim, action_dims=MGs_action_space, lr_c=1e-2, lr_a=1e-3,
                                constraint_num=1, gamma=.95, tau=.99,
                                memory_volume=buffer_size, g_ub=g_ub, lmbd_max=300,
                                writer=writer)

    c_episodes = np.zeros(EPISODES)
    c_avg = []
    g_episodes = np.zeros(EPISODES)
    g_avg = []
    lmbd_agents_ = []
    mu_g_agents = []
    voltage_off_limits = np.zeros(EPISODES)
    lmbd = np.zeros(EPISODES)
    lmbd_agents = np.zeros((EPISODES, env.MG_num))

    t = 0
    train_step = 1

    avg_span = 4
    for episode in range(EPISODES):
        time1 = time.time()
        episode_c_all = 0
        episode_g_all = 0
        start = True
        off_limits_count = 0
        for _ in range(max_episode_length):
            t += 1
            state = state_
            sequence = sequence_
            actions = agents.step(state=state, var=action_var)
            state_, sequence_, c, g = env.step(actions)
            if episode % record_interval == 0:
                lmbd = [agent.lmbd for agent in agents.agents]
                env.write_log_file(t, lmbd)
            off_limits_count += np.sum(g)
            episode_c_all += np.sum(c)
            episode_g_all += np.sum(g)
            agents.update_mu_g(g=g, step_size=1e-3 / np.log(t + 1))
            agents.store(state, actions, c, g, state_)

            if len(agents.replay_buffer.ids) >= start_train and t % train_interval == 0:
                agents.train_batch(batch_size=batch_size, critic_iter=1, actor_iter=1, train_step=train_step)
                for agent in agents.agents:
                    step_size = 1e3 / (train_step + 1e3 + 1)
                    lmbd_subgradient = agent.mu_g * env.MG_num - g_ub
                    agent.update_lmbd(step_size=step_size, subgradient=lmbd_subgradient)
                agents.lmbd_consensus()
                train_step += 1
                action_var = min(action_var_decay * action_var, action_var_min)
                agents.var = action_var
                # noise *= (1 - nosie_decay)

        c_episodes[episode] = episode_c_all / max_episode_length
        g_episodes[episode] = episode_g_all / max_episode_length
        voltage_off_limits[episode] = off_limits_count / max_episode_length
        for i, agent in enumerate(agents.agents):
            lmbd_agents[episode, i] = agent.lmbd
        # if episode % 1 == 0:
        #     print("Episode %d reward: " % episode + str(episode_reward_all))
        time2 = time.time()
        if episode % avg_span == 0 and episode > 0:
            start_i = episode - avg_span
            end_i = episode
            c_avg.append(float(np.mean(c_episodes[start_i: end_i])))
            g_avg.append(float(np.mean(g_episodes[start_i: end_i])))
            ld = [agent.lmbd for agent in agents.agents]
            mug = [agent.mu_g for agent in agents.agents]
            lmbd_agents_.append(ld)
            mu_g_agents.append(mug)
            print(
                'Episode %d to %d : Average Cost: %f    Average Off Limits: %f    Dual Var: %f    Mu_g: %f'
                '    Average Time: %f' %
                (start_i, end_i,
                 float(np.mean(c_episodes[start_i: end_i])),
                 float(np.mean(voltage_off_limits[start_i: end_i])),
                 agents.agents[0].lmbd, agents.agents[0].mu_g, time2-time1))
            writer.add_scalar('Totoal c/Average', c_avg[-1], train_step)
            writer.add_scalar('Totoal g/Average', g_avg[-1], train_step)
    c_avg = np.array(c_avg)
    g_avg = np.array(g_avg)

    TEST_EPISODES = 100
    test_c_episodes = np.zeros(TEST_EPISODES)
    test_c_avg = []
    test_g_episodes = np.zeros(TEST_EPISODES)
    test_g_avg = []
    for episode in range(TEST_EPISODES):
        episode_c_all = 0
        episode_g_all = 0
        start = True
        off_limits_count = 0
        for _ in range(max_episode_length):
            t += 1
            state = state_
            sequence = sequence_
            actions = agents.step(state=state, sequence=sequence, var=action_var)
            # actions[0][3] = max(actions[0][3], 0)
            state_, sequence_, c, g = env.step(actions)
            # if episode % record_interval == 0:
            #     env.write_log_file(t)
            off_limits_count += np.sum(g)
            episode_c_all += np.sum(c)
            episode_g_all += np.sum(g)

        test_c_episodes[episode] = episode_c_all / max_episode_length
        test_g_episodes[episode] = episode_g_all / max_episode_length
        voltage_off_limits[episode] = off_limits_count

        if episode % avg_span == 0 and episode > 0:
            start_i = episode - avg_span
            end_i = episode
            test_c_avg.append(float(np.mean(test_c_episodes[start_i: end_i])))
            test_g_avg.append(float(np.mean(test_g_episodes[start_i: end_i])))
            print(
                'Episode %d to %d : Average Cost: %f    Average Off Limits: %f' %
                (start_i, end_i,
                 float(np.mean(test_c_episodes[start_i: end_i])),
                 float(np.mean(voltage_off_limits[start_i: end_i]))))

