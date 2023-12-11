# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from chpt5.cliff_walking_env_TD import CliffWalkingEnv
from chpt5.sarsa import print_agent
import time
import random

class DynaQ:
    "在Qlearning上修改为Dyna-Q，加入环境模型model（是一个字典），每次在真实环境中收集新的数据，就把它计入到字典中"
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning #执行Q-planning的次数，对应1次Qlearning
        self.model = dict() #环境模型

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self,state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1 #将数据添加到模型中
        for _ in range(self.n_planning): #Q-planning循环
            #随机选择曾经遇到过的状态动作对
            (s, a),(r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

"Q-planning的训练函数"
def DynaQ_Cliffwalking(n_planning):
    n_col = 12
    n_row = 4
    env = CliffWalkingEnv(n_col, n_row)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(n_col, n_row, epsilon, alpha, gamma, n_planning)
    num_episodes = 300

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' %i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if(i_episode) % 10 == 0:
                    pbar.set_postfix({'episode':'%d' % (num_episodes / 10 * i+ i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list

np.random.seed(0)
random.seed(0)
n_planning_list = [0, 2, 20]
for n_planning in n_planning_list:
    print('Q-planning步数为 %d'%n_planning)
    time.sleep(0.5)
    return_list = DynaQ_Cliffwalking(n_planning)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list, label=str(n_planning) + "planning steps")
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna-Q on {}'.format('Cliff Walking'))
plt.show()
