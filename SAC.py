import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# 定义SACAgent类
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=0.001, lr_critic=0.001, lr_alpha=0.001, gamma=0.99,
                 tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic1 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)

        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.log_alpha = torch.tensor(np.log(alpha))
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.detach().numpy()
        return action

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state_batch)
            next_action = torch.tanh(next_action)
            next_target_q1 = self.target_critic1(next_state_batch, next_action)
            next_target_q2 = self.target_critic2(next_state_batch, next_action)
            next_target_q = torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + self.gamma * (1 - done_batch) * next_target_q

        current_q1 = self.critic1(state_batch, action_batch)
        current_q2 = self.critic2(state_batch, action_batch)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        sampled_actions, log_prob = self.actor(state_batch)
        sampled_actions = torch.tanh(sampled_actions)
        q1 = self.critic1(state_batch, sampled_actions)
        q2 = self.critic2(state_batch, sampled_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.log_alpha * (-log_prob - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 定义训练函数
def train_sac(env, agent, num_episodes, max_steps, batch_size):
    replay_buffer = deque(maxlen=100000)
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.train(replay_buffer, batch_size)

            if done:
                break

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])

        print(f"Episode: {episode + 1}, Total reward: {total_reward}, Avg reward (last 100 episodes): {avg_reward}")

        if avg_reward >= -200:
            print("Pendulum solved!")
            break


# 创建环境和代理
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = SACAgent(state_dim, action_dim, max_action, lr_actor=0.001, lr_critic=0.001, lr_alpha=0.001, gamma=0.99,
                 tau=0.005, alpha=0.2)

# 训练SAC算法
train_sac(env, agent, num_episodes=1000, max_steps=200, batch_size=32)