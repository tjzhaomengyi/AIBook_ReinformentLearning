# -*- coding: utf-8 -*-
__author__ = 'Mike'
import chpt2.Bandit
from chpt2.Bandit.BernoulliBandit import Solver, BernoulliBandit,plot_results
import numpy as np

"上界置信法就是选择期望奖励上界最大的动作，UCB算法在每次选择拉杆前，先估计拉动每个拉杆的期望奖励上界，" \
"使得拉动每根拉杆的期望奖励只有一个较小的概率p超过这个上界，接着选出期望奖励上界最大的拉杆，从而选择最有可能获得最大期望奖励的拉杆"
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1))) #计算上界置信
        k = np.argmax(ucb) #选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(1) #设定随机种子
K = 10
bandit_10_arm = BernoulliBandit(K)
np.random.seed(1)
coef = 1 #控制不确定性的比重系数
UBC_solver = UCB(bandit_10_arm ,coef)
UBC_solver.run(5000)
print(f"上界置信算法的累积懊悔为:{UBC_solver.regret}")
plot_results([UBC_solver],["UCB"])