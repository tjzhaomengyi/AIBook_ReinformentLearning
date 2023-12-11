# -*- coding: utf-8 -*-
__author__ = 'Mike'
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from chpt5.cliff_walking_env_TD import CliffWalkingEnv
from chpt5.sarsa import print_agent

class Qlearning:
    "Qlearning算法"
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


np.random.seed(0)
alpha = 0.1
epsilon = 0.1
gamma = 0.9
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
agent = Qlearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500 #智能体在环境中运行的序列的数量

return_list = []#记录每条记录序列的回报
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
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
            if (i_episode + 1) % 10 == 0: #每10条序列打印一下这10条序列的额平均回报
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' %np.mean(return_list[-10:])})
            pbar.update(1)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title('Q-learning on {}'. format('Cliff Walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print('Q-learning算法最终收敛得到的策略为:')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])