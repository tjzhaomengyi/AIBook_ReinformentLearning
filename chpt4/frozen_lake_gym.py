# -*- coding: utf-8 -*-
__author__ = 'Mike'
import gym
from chpt4.policy_iteration import PolicyIteraion, print_agent
from chpt4.value_iteration import ValueIteration

env = gym.make("FrozenLake-v1")
env = env.unwrapped #解封才能访问状态转移矩阵P
env.render() #环境渲染，通常是弹窗或者打印可视化环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0: #表示获得奖励为1，到达目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞索引:", holes)
print("目标索引", ends)

for a in env.P[14]:#查看目标左边一格的状态转移信息
    print(env.P[14][a])

'''
冰洞索引: {11, 12, 5, 7}
目标索引 {15}
[(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)]
[(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True)]
[(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)]
[(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False)]
四元组的表示信息(p, next_state, reward, done) ，下一个位置就是next_state
0.333表示华兴的三个方向是等概率的
'''
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteraion(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])