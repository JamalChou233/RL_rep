import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ActorCriticAgent(nn.Module):
    def __init__(self, num_states, num_actions, alpha_actor=0.001, alpha_critic=0.01, gamma=0.99):
        super(ActorCriticAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma

        self.actor = nn.Sequential(
            nn.Linear(self.num_states, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.num_actions),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.num_states, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha_critic)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.actor(state)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action

    def update(self, state, action, next_state, reward, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        # 计算TD误差
        target = reward + self.gamma * self.critic(next_state) * (1 - int(done))
        td_error = target - self.critic(state)

        # 更新Critic网络
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        probabilities = self.actor(state)
        action_prob = probabilities[0, action]
        actor_loss = -torch.log(action_prob) * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# 创建环境和Agent
env = gym.make('CartPole-v1')
agent = ActorCriticAgent(num_states=env.observation_space.shape[0], num_actions=env.action_space.n)

# 训练Agent
num_episodes = 1000
episode_list = []
reward_list = []

for episode in range(num_episodes):
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, info = env.step(action)
        agent.update(state, action, next_state, reward, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}: Total Reward = {total_reward}")
    episode_list.append(episode)
    reward_list.append(total_reward)

fig = plt.figure()
plt.plot(episode_list, reward_list)
plt.show()

# 使用训练好的Agent进行测试
state = env.reset()
state = state[0]
done = False
total_reward = 0

while not done:
    action = agent.choose_action(state)
    next_state, reward, done, _, info = env.step(action)
    state = next_state
    total_reward += reward

print(f"Test: Total Reward = {total_reward}")