# -*- coding: utf-8 -*-
__author__ = 'Mike'
"马尔可夫奖励过程"
import numpy as np

#1、计算马尔可夫回报，给定一个序列计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

#2、计算value马尔可夫奖励过程中所有状态的价值
def compute_value(P, rewards, gamma, states_num):
    "利用贝尔曼方程的举证形式计算解析解，state_num是MRP的状态数"
    rewards = np.array(rewards).reshape((-1, 1)) #将rewards写成向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

np.random.seed(0)
#定义转移概率矩阵
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
P = np.array(P)
print("-----------------------MRP马尔可夫奖励过程-------------------------")
rewards = [-1, -2, -2, 10, 1, 0] #定义奖励函数
gamma = 0.5 #定义折扣因子
#一个状态系列，s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
#1、计算马尔可夫回报
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到的回报为%s." %G) #根据本序列计算得到的回报为-2.5.
"--------------------------------------"
#计算马尔可夫过程中所有状态的价值
V = compute_value(P, rewards, gamma, 6)
print(f"MRP中每个状态价值分别为{V}")
#MRP中每个状态价值分别为[[-2.01950168] #V(S1)
 # [-2.21451846] #V(S2)
 # [ 1.16142785] #V(S3)
 # [10.53809283] #V(S4)
 # [ 3.58728554] #V(S5)
 # [ 0.        ]] #V(S6)
print("-----------------------MDP马尔可夫奖励过程-------------------------")

