from env.IEEE_33bus_4MGs_private_c import DN_33bus_4MGs
import numpy as np


if __name__ == '__main__':
    EPISODES = 100
    max_episode_length = 288
    env = DN_33bus_4MGs()
    c_episodes = np.zeros(EPISODES)
    c_avg = []
    g_episodes = np.zeros(EPISODES)
    g_avg = []
    voltage_off_limits = np.zeros(EPISODES)
    c_reg = 0.2
    t = 0

    avg_span = 4
    for episode in range(EPISODES):
        episode_c_all = 0
        episode_g_all = 0
        start = True
        off_limits_count = 0
        for _ in range(max_episode_length):
            t += 1
            # actions = np.array([[1, 0.5, 1, 0.25] for _ in range(env.MG_num)])
            actions = np.array([[0, 0, 0, 0] for _ in range(env.MG_num)])

            # actions = env.get_ref_action()
            state_, sequence_, c, private_c, g = env.step(actions)
            c = c * c_reg
            off_limits_count += np.sum(g)
            c_sum = sum(c)
            g_sum = sum(g)
            episode_c_all += c_sum
            episode_g_all += g_sum
        c_episodes[episode] = episode_c_all / max_episode_length
        g_episodes[episode] = episode_g_all / max_episode_length
        voltage_off_limits[episode] = off_limits_count
        if episode % avg_span == 0 and episode > 0:
            start_i = episode - avg_span
            end_i = episode
            c_avg.append(float(np.mean(c_episodes[start_i: end_i])))
            g_avg.append(float(np.mean(g_episodes[start_i: end_i])))
            print(
                'Episode %d to %d : Average Cost: %f    Average Off Limits: %f' %
                (start_i, end_i,
                 float(np.mean(c_episodes[start_i: end_i])),
                 float(np.mean(voltage_off_limits[start_i: end_i]))))

    c_avg = np.array(c_avg)
    g_avg = np.array(g_avg)
