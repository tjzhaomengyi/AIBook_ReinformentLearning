# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
import matplotlib.pyplot as plt
from chpt2.Bandit.BernoulliBandit import Solver, BernoulliBandit,plot_results
#贪婪算法，在每一时刻采取期望奖励估值最大的动作，EpsilonGreedy就是加了白噪声的贪婪算法
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) #随机拉动一个
        else:
            k = np.argmax(self.estimates) #选择期望最大的拉杆
        r = self.bandit.step(k) #得到本次获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


"随时间衰减的EpsilonGreedy函数"
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit,  init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K) #随机拉动一个
        else:
            k = np.argmax(self.estimates) #选择期望最大的拉杆
        r = self.bandit.step(k) #得到本次获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(1) #设定随机种子
K = 10
bandit_10_arm = BernoulliBandit(K)
np.random.seed(1)
epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver],["EpsilonGreedy"])