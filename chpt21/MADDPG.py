# -*- coding: utf-8 -*-
__author__ = 'Mike'
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import chpt7.rl_utils as rl_utils

"我们使用的环境为多智能体粒子环境MPE，它是一些面向多智能体交互的环境的集合，在这个环境中，粒子智能体可以移动、通信、看到其他智能体，也可以和固定位置额地标交互"
import gym
import sys
# sys.path.append("multiagent_particle_envs")
"之前gym版本为0.18.3"
from chpt21.multiagent_particle_envs.multiagent.environment import MultiAgentEnv
import chpt21.multiagent_particle_envs.multiagent.scenarios as scenarios


def make_env(scenario_name):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

"本章选择MPE中的simple_adversary环境作为示例，该环境中有1个红色的对抗智能体adversary，N个蓝色的整场智能体，以及N个地点(一般N=2)，" \
"这N个地点中有一个是目标地点（绿色）。这N个正常智能体知道哪一个是目标地点，但对抗智能体不知道。正常智能体是合作关系：它们其中任意一个距离目标地点足够近，则每个正常智能体" \
"都能获得相同的奖励。对抗智能体如果距离目标地点足够近，也能获得奖励，但是它需要猜哪一个才是目标地点。因此，正常智能体需要进行合作，分散到不同的坐标点，以前对抗智能体"
"MPE环境中的每个智能体的动作空间是离散的。DDPG算法本身需要使智能体的动作对于其策略参数可导，这对连续的动作空间来说是成立的，但是对于离散的动作" \
"空间并不成立。但这并不意味着当前的任务不能使用MADDPG算法求解，因为我们可以使用一个叫做Gumbel-softmax的方法来得到离散分布的近似采样"
def onehot_from_logits(logits, eps=0.01):
    "生成最优动作的one-hot形式"
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    #生成随机动作，转换成one-hot形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False).to(logits.device)
    #通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i,r in enumerate(torch.rand(logits.shape[0]))])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    "从Gumbel(0,1)分布中采样"
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    "从Gumbel-Softmax分布中采样"
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    # logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0):
    "从Gumbel-Softmax分布中采样，并进行离散化"
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    #返回一个y_hard的one-hot编码，但是它的梯度是y，我们既能够得到一个与环境交互的离散动作，又可以正确地反传梯度
    return y

#实现单智能体DDPG，包含Actor-Critic网络，以及计算动作的函数
"这是一个简单的两层神经网络"
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    "DDPG算法"
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim, actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)

        #初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        #初始化目标策略网络并设置和策略网络相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device


    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]


    def soft_update(self, net, target_net, tau):
        #软更新
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


"MADDPG"
class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents= []
        for i in range(len(env.agents)):
            self.agents.append(DDPG(state_dims[i], action_dims[i], critic_input_dim, hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        # states = [torch.tensor([states[i]], dtype=torch.float, device=self.device)
        #           for i in range(len(env.agents))]
        # return [agent.take_action(state,explore) for agent, state in zip(self.agents, states)]

        states = [
        torch.tensor([states[i]], dtype=torch.float, device=self.device)
        for i in range(len(env.agents))
        ]
        return [agent.take_action(state, explore) for agent, state in zip(self.agents, states)]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1- done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())

        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []

        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()


    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)




def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    #对学习的策略进行评估，不会进行探索
    env= make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

num_episodes = 5000
episode_length = 25 #每条序列的最大长度
buffer_size = 100000
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 4000

env_id = "simple_adversary"
env = make_env(env_id)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n)
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0])

critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)

return_list = []  # 记录每一轮的回报（return）
total_step = 0
for i_episode in range(num_episodes):
    state = env.reset()
    # ep_returns = np.zeros(len(env.agents))
    for e_i in range(episode_length):
        actions = maddpg.take_action(state, explore=True)
        next_state, reward, done, _ = env.step(actions)
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state

        total_step += 1
        if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

            sample = [stack_array(x) for x in sample]
            for a_i in range(len(env.agents)):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 100 == 0:
        ep_returns = evaluate(env_id, maddpg, n_episode=100)
        return_list.append(ep_returns)
        print(f"Episode:{i_episode + 1},{ep_returns}")

print(return_list)
# [[-104.22065558509424, 12.439860795330793, 12.439860795330793], [-31.337669019934264, 2.6696172102807303, 2.6696172102807303], [-19.343491132240526, -15.849744400511558, -15.849744400511558], [-14.741427081960795, -3.78848122185604, -3.78848122185604], [-12.426771644217563, -2.509171944683219, -2.509171944683219], [-13.001877856564047, 3.5842439219783095, 3.5842439219783095], [-12.046590955141932, 3.3277130154793415, 3.3277130154793415], [-11.99086885310488, 4.573482757389915, 4.573482757389915], [-11.253579914649432, 6.304561874169534, 6.304561874169534], [-10.88992326362551, 7.844801232461369, 7.844801232461369], [-10.159626091978744, 7.067212632678632, 7.067212632678632], [-8.807281354325141, 6.21417707063183, 6.21417707063183], [-9.007234227903872, 6.62152223360061, 6.62152223360061], [-8.721307664825003, 6.1040348362803565, 6.1040348362803565], [-7.016801848291476, 4.92294213450325, 4.92294213450325], [-8.173656261603044, 5.333391630714142, 5.333391630714142], [-8.994287744468886, 5.77722125535756, 5.77722125535756], [-8.254991403259403, 5.379775935452114, 5.379775935452114], [-9.63687842494737, 5.889185246974522, 5.889185246974522], [-7.2086550041788096, 4.834475254383641, 4.834475254383641], [-9.498829541482602, 5.600178878197573, 5.600178878197573], [-9.046924394915225, 5.944425746911233, 5.944425746911233], [-8.827102377163783, 5.201825313599244, 5.201825313599244], [-7.88218311557143, 5.186197442304333, 5.186197442304333], [-8.49768985291731, 5.452996911217176, 5.452996911217176], [-9.78754912592921, 5.17353437148085, 5.17353437148085], [-7.994229256648374, 4.428998396458902, 4.428998396458902], [-8.21412942198809, 4.8241171211535505, 4.8241171211535505], [-7.677227870696316, 4.487838248607318, 4.487838248607318], [-7.365377116254546, 4.478078270403421, 4.478078270403421], [-6.918740215609912, 4.217683817498419, 4.217683817498419], [-8.206354943146906, 5.1495942812903746, 5.1495942812903746], [-7.740561219432013, 4.69733880626667, 4.69733880626667], [-7.581009892471343, 4.3928033533510105, 4.3928033533510105], [-7.866665592207869, 4.573777835719444, 4.573777835719444], [-7.904983836451992, 4.649869278336736, 4.649869278336736], [-8.177737490034527, 4.250796914488585, 4.250796914488585], [-7.235251968020889, 4.402419030297032, 4.402419030297032], [-8.057835704140013, 4.3963195685383685, 4.3963195685383685], [-8.003692599982195, 3.738309657816671, 3.738309657816671], [-9.140861715514095, 4.848176886634885, 4.848176886634885], [-8.261887580529276, 4.045491161086763, 4.045491161086763], [-9.571828126782723, 5.20072577062457, 5.20072577062457], [-7.417729111835252, 4.13101420606668, 4.13101420606668], [-8.641259582310974, 4.777926129037788, 4.777926129037788], [-8.138848696049457, 4.341546638026829, 4.341546638026829], [-9.720515423292854, 5.2484564822880735, 5.2484564822880735], [-10.272041657378447, 5.377403999193589, 5.377403999193589], [-11.302821808252313, 6.24311230056548, 6.24311230056548], [-9.646472960097215, 5.057438164503499, 5.057438164503499]]

return_array = np.array(return_list)
# return_array = np.array(
#     [[-104.22065558509424, 12.439860795330793, 12.439860795330793], [-31.337669019934264, 2.6696172102807303, 2.6696172102807303], [-19.343491132240526, -15.849744400511558, -15.849744400511558], [-14.741427081960795, -3.78848122185604, -3.78848122185604], [-12.426771644217563, -2.509171944683219, -2.509171944683219], [-13.001877856564047, 3.5842439219783095, 3.5842439219783095], [-12.046590955141932, 3.3277130154793415, 3.3277130154793415], [-11.99086885310488, 4.573482757389915, 4.573482757389915], [-11.253579914649432, 6.304561874169534, 6.304561874169534], [-10.88992326362551, 7.844801232461369, 7.844801232461369], [-10.159626091978744, 7.067212632678632, 7.067212632678632], [-8.807281354325141, 6.21417707063183, 6.21417707063183], [-9.007234227903872, 6.62152223360061, 6.62152223360061], [-8.721307664825003, 6.1040348362803565, 6.1040348362803565], [-7.016801848291476, 4.92294213450325, 4.92294213450325], [-8.173656261603044, 5.333391630714142, 5.333391630714142], [-8.994287744468886, 5.77722125535756, 5.77722125535756], [-8.254991403259403, 5.379775935452114, 5.379775935452114], [-9.63687842494737, 5.889185246974522, 5.889185246974522], [-7.2086550041788096, 4.834475254383641, 4.834475254383641], [-9.498829541482602, 5.600178878197573, 5.600178878197573], [-9.046924394915225, 5.944425746911233, 5.944425746911233], [-8.827102377163783, 5.201825313599244, 5.201825313599244], [-7.88218311557143, 5.186197442304333, 5.186197442304333], [-8.49768985291731, 5.452996911217176, 5.452996911217176], [-9.78754912592921, 5.17353437148085, 5.17353437148085], [-7.994229256648374, 4.428998396458902, 4.428998396458902], [-8.21412942198809, 4.8241171211535505, 4.8241171211535505], [-7.677227870696316, 4.487838248607318, 4.487838248607318], [-7.365377116254546, 4.478078270403421, 4.478078270403421], [-6.918740215609912, 4.217683817498419, 4.217683817498419], [-8.206354943146906, 5.1495942812903746, 5.1495942812903746], [-7.740561219432013, 4.69733880626667, 4.69733880626667], [-7.581009892471343, 4.3928033533510105, 4.3928033533510105], [-7.866665592207869, 4.573777835719444, 4.573777835719444], [-7.904983836451992, 4.649869278336736, 4.649869278336736], [-8.177737490034527, 4.250796914488585, 4.250796914488585], [-7.235251968020889, 4.402419030297032, 4.402419030297032], [-8.057835704140013, 4.3963195685383685, 4.3963195685383685], [-8.003692599982195, 3.738309657816671, 3.738309657816671], [-9.140861715514095, 4.848176886634885, 4.848176886634885], [-8.261887580529276, 4.045491161086763, 4.045491161086763], [-9.571828126782723, 5.20072577062457, 5.20072577062457], [-7.417729111835252, 4.13101420606668, 4.13101420606668], [-8.641259582310974, 4.777926129037788, 4.777926129037788], [-8.138848696049457, 4.341546638026829, 4.341546638026829], [-9.720515423292854, 5.2484564822880735, 5.2484564822880735], [-10.272041657378447, 5.377403999193589, 5.377403999193589], [-11.302821808252313, 6.24311230056548, 6.24311230056548], [-9.646472960097215, 5.057438164503499, 5.057438164503499]]
# )

for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
    plt.figure()
    plt.plot(
        np.arange(return_array.shape[0]) * 100,
        rl_utils.moving_average(return_array[:, i], 9))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"{agent_name} by MADDPG")
plt.show()