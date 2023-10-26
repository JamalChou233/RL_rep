import torch
import torch.nn as nn
import torch.optim as optim
import gym


# 定义策略网络
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


# 定义PPO-Penalty算法
def ppo_penalty(env_name, num_epochs, num_steps, epsilon, target_kl, beta):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = Policy(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        for step in range(num_steps):
            state = torch.FloatTensor(state)
            action_probs, value = policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _, info = env.step(action.item())
            total_reward += reward

            next_state = torch.FloatTensor(next_state)
            _, next_value = policy(next_state)

            advantage = reward + (1 - done) * next_value - value

            log_prob = dist.log_prob(action)
            # print(f"log_prob: {log_prob}")
            old_action_probs = action_probs.detach()
            old_dist = torch.distributions.Categorical(old_action_probs)
            old_log_prob = old_dist.log_prob(action)

            ratio = torch.exp(log_prob - old_log_prob)
            kl = (old_log_prob - log_prob).mean()
            penalty = torch.max(torch.zeros(1), kl - 2.0 * target_kl) ** 2
            surrogate_loss = -torch.min(ratio * advantage,
                                        torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage - beta * penalty)

            optimizer.zero_grad()
            surrogate_loss.mean().backward()
            optimizer.step()

            state = next_state.numpy()

            if done:
                break

        print(f"Epoch: {epoch}, Total reward: {total_reward}")

    env.close()


# 运行PPO-Penalty算法
ppo_penalty('CartPole-v1', num_epochs=100, num_steps=200, epsilon=0.2, target_kl=0.01, beta=0.5)