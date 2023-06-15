from rl_torch.cmdp_onpolicy_spread10 import multi_agent_system
import torch
from rl_torch.nn import Actor, Critic, Actor_state, Critic_state
import numpy as np
from copy import copy
import random
import matplotlib.pyplot as plt
import multiagent.scenarios as scenarios
from multiagent.environment_cmdp_target import MultiAgentEnv_cmdp_target
import time

EPISODES = int(8e4)
max_episode_length = 25
scenario_name = 'simple_spread_cmdp_target_10_agents'
load_model = False
reward_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
batch_size = 1024
start_train = 25
train_interval = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g_ub = 2.5
w6 = [[0.99, 0.005, 0, 0, 0, 0, 0, 0, 0, 0.005],
      [0.005, 0.99, 0.005, 0, 0, 0, 0, 0, 0, 0],
      [0, 0.005, 0.99, 0.005, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.005, 0.99, 0.005, 0, 0, 0, 0, 0],
      [0, 0, 0, 0.005, 0.99, 0.005, 0, 0, 0, 0],
      [0, 0, 0, 0, 0.005, 0.99, 0.005, 0, 0, 0],
      [0, 0, 0, 0, 0, 0.005, 0.99, 0.005, 0, 0],
      [0, 0, 0, 0, 0, 0, 0.005, 0.99, 0.005, 0],
      [0, 0, 0, 0, 0, 0, 0, 0.005, 0.99, 0.005],
      [0.005, 0, 0, 0, 0, 0, 0, 0, 0.005, 0.99]]
w7 = np.eye(10)
lmbd_max = 300
scenario = scenarios.load(scenario_name + ".py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv_cmdp_target(world, scenario.reset_world, scenario.reward, scenario.constraint, scenario.observation)
agent_num = env.n
action_dim = [env.action_space[i].n for i in range(agent_num)]
obs_shape = env.observation_space[0].shape[0]
state = env.reset()[0]
state_dim = len(state)
c_episode = []
g_episode = []
summary_dir = 'tensorboard_log/w6_3/'
# w = np.eye(10)

agents = multi_agent_system(agent_num=env.n, device=device, communication_matrix=w6,
                            actor_fn=Actor_state, critic_fn=Critic_state, state_dim=state_dim,
                            sequence_dim=0, action_dims=action_dim, lr_c=1e-2, lr_a=1e-3,
                            constraint_num=1, gamma=.95, tau=.99, memory_volume=int(1e6),
                            action_min=np.zeros((env.n, 5)), action_max=np.ones((env.n, 5)),
                            g_ub=g_ub, lmbd_max=lmbd_max, summary_dir=summary_dir)
for agent in agents.agents:
    agent.lmbd = np.random.uniform(0, 5)

Rewards = np.zeros(shape=(EPISODES, env.n))
Rewards_Avg = []
Constraints = np.zeros(EPISODES)
Constraints_Avg = []
Lmbd_agents_ = []
mu_g_agents = []
Collisions = np.zeros(EPISODES)
Lmbd = np.zeros(EPISODES)
Lmbd_agents = np.zeros((EPISODES, agent_num))
# if load_model == True:
#     saver.restore(sess, ckpt_path)
t = 0
train_step = 1
obs_ = None
# noise = 0.
# nosie_decay = 0
avg_span = 100
for episode in range(EPISODES):
    episode_reward_all = np.zeros(agent_num)
    episode_cosntraint_all = 0
    start = True
    collision_count = 0
    for _ in range(max_episode_length):
        t += 1

        if start == True:
            obs = env.reset()[0]
            start = False
        else:
            obs = obs_
        # time1 = time.time()
        actions = agents.step(state=obs)
        # for i in range(len(actions)):
            # actions[i] = actions[i][0]
        # time2 = time.time()
        obs_, c, g, terminal, info = env.step(actions)
        obs_ = obs_[0]
        for i in range(agent_num):
            c[i] *= reward_weight[i]
        collision_count += np.sum(g) / 2
        # print(time2 - time1)
        # env.render()
        episode_reward_all += c
        episode_cosntraint_all += np.mean(g)
        agents.update_mu_g(g=g, step_size=1e-3/np.log(t+1))
        agents.store(obs, actions, c, g, obs_)

        if len(agents.replay_buffer.ids) >= batch_size * start_train and t % train_interval == 0:
            # agents.get_batch(batch_size=batch_size)
            agents.train_batch(batch_size=batch_size, critic_iter=1, actor_iter=1, train_step=train_step)
            tt = np.log(train_step + 1)
            # tt = train_step
            for agent in agents.agents:
                step_size = 1e3 / (train_step + 1e3 + 1)
                lmbd_subgradient = agent.mu_g * 25 * agent_num / 2 - g_ub
                agent.update_lmbd(step_size=step_size, subgradient=lmbd_subgradient)
            agents.lmbd_consensus()
            train_step += 1
            # noise *= (1 - nosie_decay)
        if terminal[0] == True:
            break

    Rewards[episode, :] = episode_reward_all
    Constraints[episode] = episode_cosntraint_all / max_episode_length
    Collisions[episode] = collision_count
    for i, agent in enumerate(agents.agents):
        Lmbd_agents[episode, i] = agent.lmbd
    # if episode % 1 == 0:
    #     print("Episode %d reward: " % episode + str(episode_reward_all))
    if episode % avg_span == 0 and episode > 0:
        start_i = episode - avg_span
        end_i = episode
        Rewards_Avg.append(float(np.mean(Rewards[start_i: end_i])))
        Constraints_Avg.append(float(np.mean(Collisions[start_i: end_i])))
        ld = [agent.lmbd for agent in agents.agents]
        mug = [agent.mu_g for agent in agents.agents]
        Lmbd_agents_.append(ld)
        mu_g_agents.append(mug)
        print(
            'Episode %d to %d : Average Reward: %f    Average Collisions: %f    Dual Var: %f %f %f    Mu_g: %f %f %f' %
            (start_i, end_i,
             float(np.mean(Rewards[start_i: end_i])),
             float(np.mean(Collisions[start_i: end_i])),
             agents.agents[0].lmbd, agents.agents[1].lmbd, agents.agents[2].lmbd,
             agents.agents[0].mu_g, agents.agents[1].mu_g, agents.agents[2].mu_g))
Rewards_Avg = np.array(Rewards_Avg)
Constraints_Avg = np.array(Constraints_Avg)



# while True:
#     obs = env.reset()[0]
#     for _ in range(25):
#         actions = agents.step(obs)
#         obs_, reward, constraint, terminal, info = env.step(actions)
#         env.render()
#         obs = obs_[0]
#         time.sleep(0.1)
