# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
import copy
from chpt4.cliff_walk import CliffWalkingEnv
class PolicyIteraion:
    "策略迭代算法：策略提升公式：π'(s)=arg maxQπ(s,a) = argmax{r(s, a) + γ∑P(s'|s,a)Vπ(s')}"
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow #初始化一个(nrow, ncol)的0矩阵
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)] #初始化随机均匀策略随机
        self.theta = theta #测录评估收敛阈值
        self.gamma = gamma #折扣因子

    def policy_evaluation(self): #策略评估
        cnt = 1 #计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] #开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        #本章环境比较特殊，奖励和下一个状态有关，所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list) #状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break #满足收敛条件，退出迭代
            cnt += 1
        print(f"策略评估进行{cnt}轮后完成。")


    def policy_imporvement(self): #策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) #计算有几个动作得到了最大的Q值
            #让这些动作均分概率
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self): #策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_imporvement()
            if old_pi == new_pi:
                break


#打印当前策略在每个状态下的价值以及只能体会采取的动作。^o<o表示等概率采取向上和向左两种动作，ooo>表示在当前状态下只采取向右动作
def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            #一些特殊的状态，例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('*****', end=" ")
            elif(i * agent.env.ncol + j) in disaster:
                print("EEEE", end= ' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=" ")
    print()


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteraion(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47]) #策略评估进行51轮后完成。