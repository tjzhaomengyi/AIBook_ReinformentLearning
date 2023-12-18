# -*- coding: utf-8 -*-
__author__ = 'Mike'
import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import chpt7.rl_utils as rl_utils
import copy
"对于策略网络和价值网络，我们都采用只有一层隐藏层的神经网络。策略网络的输出层用正切函数y=tanhx作为激活函数，正切函数的值域是[-1,1]，方便按比例调整成" \
"环境可以接受的动作范围。在DDPG中处理的是与连续动作交互的环境，Q网络的输入是和状态和动作拼接后的向量，Q网络输出是一个值，表示该状态动作对的价值"

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 =torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = actions_bound #action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

#AC网络多一个价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) #拼接状态和动作
        x = F.relu(self.fc1(cat))
        return self.fc2(x)

"这是一个简单的两层神经网络"
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x:x):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x

"在策略网络采取动作的时候，为了更好地探索，我们向动作中加入高斯噪声。添加的噪声符合奥恩斯坦-乌伦贝克随机过程（OU随机过程），OU随机过程是与时间相关的，适用于有惯性的系统。" \
"相比如DDPG中常用的正态分布噪声系统，正态分布噪声更加简单"
class DDPG:
    "DDPG算法"
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim, discrete, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        out_fn = (lambda  x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim, activation=F.relu, out_fn=out_fn).to(device)
        self.target_actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim, activation=F.relu, out_fn=out_fn).to(device)
        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        #初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        #初始化目标策略网络并设置和策略网络相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma #高斯噪声的标准差，均值直接设置为0
        self.action_bound = action_bound #action_bound是环境可以接受的动作最大值
        self.tau = tau #目标网络更新参数
        self.action_dim = num_out_actor
        self.device = device


    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        #给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        #软更新
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device) #注意这里是连续动作
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(torch.cat([states, actions], dim=1)),q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(torch.cat([states, self.actor(states)],dim=1))) #策略网络就是为了使得Q值最大化
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor) #软更新策略网络
        self.soft_update(self.critic, self.target_critic) #软更新价值网络



num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau=0.005 #软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01 #动作的高斯噪声
actor_lr = 5e-4
critic_lr = 5e-3
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "Pendulum-v0"
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0] #动作最大值
agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size) #离线策略

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG Continuous on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG Continuous on {}'.format(env_name))
plt.show()