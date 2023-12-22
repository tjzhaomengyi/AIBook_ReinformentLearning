# -*- coding: utf-8 -*-
__author__ = 'Mike'
"多智能体环境：ma_gym库中的combat环境。combat是一个在二维的额格子世界上进行额两个队伍的对战模拟游戏，每个智能体的动作集合为：向四周移动1格，" \
"攻击周围3*3格范围内其他敌对智能体，或者不采取任何行动。起初每个智能体有3个生命值，如果智能体在对人的攻击范围内被攻击到，损失1点生命值，生命掉到0死亡，最后存活队伍获胜。" \
"每个智能体的攻击有一轮的冷却时间。" \
"在游戏中，我们能够控制一个队伍的所有智能体与另一个队伍的智能体进行对战。另一个队伍的智能体使用固定的算法：攻击在范围内最近的敌人，如果攻击范围内没有敌人，则向敌人靠近。" \
"" \
""
import torch
import torch.nn.functional as F
import numpy as np
import chpt7.rl_utils as rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("./magym")
from chpt20.magym.ma_gym.envs.combat.combat import Combat
# from ma_gym.envs.combat.combat import Combat



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


#AC网络多一个价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device):
        # 策略网络参数不需要优化更新器
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.eps = eps #PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) #advantage优势函数的截断
        actor_loss = torch.mean(-torch.min(surr1, surr2)) #PPO损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


"IPPO代码主要部分，在使用时使用了参数共享的技巧，即对于所有智能体使用同一套策略参数，这样做的好处是湿的模型训练数据更多，" \
"同时训练更稳定。能这样做的前提是，两个智能体时同质的，即它们的状态空间和动作空间是完全一致的，并且它们的优化目标也完全一致。" \
"这里不再展示智能体的回报，而是将IPPO训练的两个智能体团队的胜率作为实验结果"
actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 100000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

team_size = 2
grid_size = (15, 15)
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
#两个智能体共享一个策略
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/ 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_dict_1 = {'states':[], 'actions':[], 'next_states':[], 'rewards':[], 'dones':[]}
            transition_dict_2 = {'states':[], 'actions':[], 'next_states':[], 'rewards':[], 'dones':[]}
            s = env.reset()
            terminal = False
            while not terminal:
                a_1 = agent.take_action(s[0])
                a_2 = agent.take_action(s[1])
                next_s, r, done, info = env.step([a_1, a_2])
                transition_dict_1['states'].append(s[0])
                transition_dict_1['actions'].append(a_1)
                transition_dict_1['next_states'].append(next_s[0])
                transition_dict_1['rewards'].append(r[0] + 100 if info['win'] else r[0]-0.1)
                transition_dict_1['dones'].append(False)

                transition_dict_2['states'].append(s[1])
                transition_dict_2['actions'].append(a_2)
                transition_dict_2['next_states'].append(next_s[1])
                transition_dict_2['rewards'].append(r[1] + 100 if info['win'] else r[1] - 0.1)
                transition_dict_2['dones'].append(False)

                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            agent.update(transition_dict_1)
            agent.update(transition_dict_2)
            if (i_episode+1) % 100 == 0:
                pbar.set_postfix({'episode':'%d' %(num_episodes/ 10 * i + i_episode + 1),'return':'%.3f' % np.mean(win_list[-100:])})
            pbar.update(1)

#胜率结果图
win_array = np.array(win_list)
#每100条轨迹取一次平均
win_array = np.mean(win_array.reshape(-1, 100), axis=1)

episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Win rate')
plt.title('IPPO on Combat')
plt.show()