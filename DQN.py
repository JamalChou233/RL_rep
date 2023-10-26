import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        q_values = self.output(x)
        return q_values


# 定义DQN算法
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        q_values = self.model(state_batch)
        next_q_values = self.target_model(next_state_batch)

        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# 定义训练函数
def train_dqn(env, agent, num_episodes, max_steps, batch_size):
    replay_buffer = []
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.train(replay_buffer, batch_size)

            if done:
                break

        agent.update_target_model()

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])

        print(f"Episode: {episode + 1}, Total reward: {total_reward}, Avg reward (last 100 episodes): {avg_reward}")

        if avg_reward >= 195:
            print("CartPole solved!")
            break


# 创建环境和代理
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995)

# 训练DQN算法
train_dqn(env, agent, num_episodes=1000, max_steps=200, batch_size=32)