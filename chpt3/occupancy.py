# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
from chpt3.montecalo import sample, MDP, Pi_1, Pi_2
def occupancy(episodes, s, a, timestep_max, gamma):
    "计算状态动作对（s,a）出现的频率，【目的】以此来估算策略的占用量"
    rho = 0
    total_times = np.zeros(timestep_max) #记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max) #记录（s_t, a_t） = (s, a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2) #0.12001103225828164 0.23313446679404293
#通过以上结果发现，不同策略对于同一个撞他及动作对的占用量是不一样的