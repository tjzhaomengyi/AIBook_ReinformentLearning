# -*- coding: utf-8 -*-
__author__ = 'Mike'
"Thompson采样法是先假设拉动每根拉杆的奖励服从一个特定的概率分布，然后根据拉动每根拉杆的期望奖励来进行选择。但是由于计算所有拉杆的期望奖励的" \
"代价比较高，Thompson Sampling算法采用采样的方式，即根据当前每个动作a的奖励概率分布进行一轮采样，得到一组各根拉杆的奖励样本，在选择样本中奖励" \
"最大的动作，可以看出Thompson Sampling是一种计算所有拉杆的最高奖励概率的蒙特卡洛采样方法"
"怎样得到当前每个动作a的奖励概率分布并且在过程中进行更新？我们通常使用Beta分布对当前每个动作的奖励进行分布建模，即，若某拉杆被选择了k次，其中m1" \
"次奖励为1， m2次奖励为0，则拉杆的奖励服从参数为(m1 + 1,m2 + 1)的Beta分布。"
import chpt2.Bandit
from chpt2.Bandit.BernoulliBandit import Solver, BernoulliBandit,plot_results
import numpy as np



class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) #表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K) #表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b) #按照beta分布采样一组奖励样本
        k = np.argmax(samples) #选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r #更新Beta分布的第一个参数
        self._b[k] += (1 - r) #更新Beat分布的第二个参数
        return k

np.random.seed(1) #设定随机种子
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"随机生成了一个{K}个臂老虎机")
print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}号，其概率为{bandit_10_arm.best_prob:.4f}")
np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print(f"汤普森采样算法的累积懊悔为：{thompson_sampling_solver.regret}")
plot_results([thompson_sampling_solver], ["ThompsonSampling"])