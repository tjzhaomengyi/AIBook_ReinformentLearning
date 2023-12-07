# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
import matplotlib.pyplot as plt

'''伯努利方法解决多臂老虎机，就是让一个杆子多次尝试，然后获得这个杆子的期望奖励'''
class BernoulliBandit:
    #注意类中大小K的区别，K表示拉杆个数，k表示选中的第k号拉杆
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs) #从probs中获得获奖概率最大的
        self.best_prob = self.probs[self.best_idx] #最大的获奖概率
        self.K = K

    def step(self, k):
        #当获奖玩家选择了k号拉杆后，根据拉动该k号拉杆获得奖励的概率返回1获奖或者0未获奖
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

#给出求解方案
class Solver:
    '''多臂老虎机算法基本框架'''
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0 #当前步子的累积懊悔
        self.actions = [] #维护一个列表，记录每步动作
        self.regrets = [] #记录每一步的懊悔

    def update_regret(self, k):
        #计算累积懊悔并保存，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    #不同的策略run_one_step不同
    def run_one_step(self):
        #返回当前动作选择哪一个拉杆，由每个具体的策略实现
        raise NotImplemented

    def run(self, num_steps):
        #运行一定次数，num_steps为总运行次数
        for _ in  range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

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

def plot_results(solvers, solver_names):
    "生成累积懊悔岁时间变化的图像，输入solvers是一个列表，列表中的每个元素是一种特定的策略"
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(f"{solvers[0].bandit.K} - armed bandit")
    plt.legend()
    plt.show()

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



# np.random.seed(1) #设定随机种子
# K = 10
# bandit_10_arm = BernoulliBandit(K)
# print(f"随机生成了一个{K}个臂老虎机")
# print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}号，其概率为{bandit_10_arm.best_prob:.4f}")
#
# np.random.seed(1)
# epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
# epsilon_greedy_solver.run(5000)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
# plot_results([epsilon_greedy_solver],["EpsilonGreedy"])