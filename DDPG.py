import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ref: https://blog.csdn.net/b_b1949/article/details/128997146

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义DDPGAgent类
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor, lr_critic, gamma, tau):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

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

        target_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic(next_state_batch, target_actions)
        target_q_values = reward_batch + self.gamma * target_q_values * (1 - done_batch)

        q_values = self.critic(state_batch, action_batch)

        critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 定义训练函数
def train_ddpg(env, agent, num_episodes, max_steps, batch_size):
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
agent = DDPGAgent(state_dim, action_dim, max_action, lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.001)

# 训练DDPG算法
train_ddpg(env, agent, num_episodes=1000, max_steps=200, batch_size=32)