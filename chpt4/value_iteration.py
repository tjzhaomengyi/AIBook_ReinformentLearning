# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
from chpt4.cliff_walk import CliffWalkingEnv
from chpt4.policy_iteration import print_agent

class ValueIteration:
    "价值迭代算法"
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow #初始化价值为0
        self.theta = theta #价值收敛值
        self.gamma = gamma
        #价值迭代后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] #开始状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)#【注意】这一行和下一行代码是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break #满足收敛条件
            cnt += 1
        print(f"价值迭代一共进行{cnt}轮")
        self.get_policy()

    def get_policy(self):#根据价值函数到处一个贪婪策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += r + p * self.gamma * self.v[next_state] * (1 - done)
                qsa_list.append(res)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) #计算有几个动作得到了最大的Q值
            #让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47]) #价值迭代一共进行4轮a