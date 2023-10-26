import torch
import torch.nn as nn
import torch.optim as optim
import gym


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


# 定义PPO-Clip算法
def ppo_clip(env, model, epochs, batch_size, epsilon, gamma, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_probs_old, value_old = model(torch.FloatTensor(state))
            action_dist_old = torch.distributions.Categorical(action_probs_old)
            action_old = action_dist_old.sample()

            next_state, reward, done, _ = env.step(action_old.item())
            total_reward += reward

            action_probs, value = model(torch.FloatTensor(next_state))
            action_dist = torch.distributions.Categorical(action_probs)

            ratio = torch.exp(action_dist.log_prob(action_old) - action_dist_old.log_prob(action_old))
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

            advantage = reward + gamma * value - value_old

            actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            critic_loss = loss_fn(value, reward + gamma * value)

            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print('Epoch: {}, Total Reward: {}'.format(epoch + 1, total_reward))


# 创建环境和模型
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 64
model = ActorCritic(state_dim, action_dim, hidden_dim)

# 运行PPO-Clip算法
epochs = 100
batch_size = 32
epsilon = 0.2
gamma = 0.99
lr = 0.001
ppo_clip(env, model, epochs, batch_size, epsilon, gamma, lr)