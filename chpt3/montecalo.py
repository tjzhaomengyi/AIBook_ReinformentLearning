# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
S = ["s1", "s2", "s3", "s4", "s5"]#状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]#工作集合

#状态转移函数
P = {
    "s1-保持s1-s1":1.0, "s1-前往s2-s2":1.0,
    "s2-前往s1-s1":1.0, "s2-前往s3-s3":1.0,
    "s3-前往s4-s4":1.0, "s3-前往s5-s5":1.0,
    "s4-前往s5-s5":1.0, "s4-概率前往-s2":0.2,
    "s4-概率前往-s3":0.4, "s4-概率前往-s4":0.4
}

#奖励函数
R = {
    "s1-保持s1":-1, "s1-前往s2":0,
    "s2-前往s1":-1, "s2-前往s3":-2,
    "s3-前往s4":-2, "s3-前往s5":0,
    "s4-前往s5":10, "s4-概率前往":1
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)
#策略1，随机策略
Pi_1 = {
    "s1-保持s1":0.5, "s1-前往s2":0.5,
    "s2-前往s1":0.5, "s2-前往s3":0.5,
    "s3-前往s4":0.5, "s3-前往s5":0.5,
    "s4-前往s5":0.5, "s4-概率前往":0.5
}

#策略2
Pi_2 = {
    "s1-保持s1":0.6, "s1-前往s2":0.4,
    "s2-前往s1":0.3, "s2-前往s3":0.7,
    "s3-前往s4":0.5, "s3-前往s5":0.5,
    "s4-前往s5":0.1, "s4-概率前往":0.9
}

#把输入的两个两个字符通过"-"连接，便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


#2、计算value马尔可夫奖励过程中所有状态的价值
def compute_value(P, rewards, gamma, states_num):
    "利用贝尔曼方程的举证形式计算解析解，state_num是MRP的状态数"
    rewards = np.array(rewards).reshape((-1, 1)) #将rewards写成向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

#我们定义一个采样函。采样函数需要遵守状态转移矩阵和相应的策略，每次将(s,a,r,s_next)元组放入序列中，直到到达终止序列。
def sample(MDP, Pi, timestep_max, number):
    "采样函数，策略pi，限制最长时间步timestep_max,总共采样序列数number"
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]#随机选择一个除s5以外的状态s作为起点
        #当前状态为终止状态或者时间步太长时，一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            #在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            #根据状态转移概率得到下一个状态s_next
            # s_next = 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next)) #把元组放入序列中
            s = s_next #为下次循环准备
        episodes.append(episode)
    return episodes

#对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1): #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

#采样五次，每个序列最大长度不超过20步
gamma = 0.5
MDP = S, A, P, R, gamma
timestep_max = 20
#采样1000次
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma= 0.5
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
MC(episodes, V, N, gamma)
print(f"使用蒙特卡洛方法计算MDP的状态价值为\n{V}")



# 使用蒙特卡洛方法计算MDP的状态价值为
#  {'s1': -1.223325902209694, 's2': -1.6905379469031547, 's3': 0.4821945861250693, 's4': 5.957078598478785, 's5': 0}
'''和使用MDP计算的结果基本一致'''