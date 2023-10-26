import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actor_output = torch.softmax(self.actor(x), dim=-1)
        critic_output = self.critic(x)
        return actor_output, critic_output


def ppo_penalty(env_name, lr=0.001, gamma=0.99, eps_clip=0.2, penalty_coeff=0.1, max_episodes=1000, max_steps=1000):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    model = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(max_episodes):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0
        t = 0

        while not done and t < max_steps:
            t += 1

            action_probs, value = model(torch.FloatTensor(state))
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, info = env.step(action.item())

            total_reward += reward

            next_action_probs, _ = model(torch.FloatTensor(next_state))
            next_action_dist = Categorical(next_action_probs)
            next_action_log_prob = next_action_dist.log_prob(action)

            advantage = reward + gamma * (1 - done) * model(torch.FloatTensor(next_state))[1].item() - value.item()

            ratio = torch.exp(log_prob - next_action_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

            actor_loss = -torch.min(surr1, surr2)
            critic_loss = advantage ** 2

            penalty = torch.abs(action_probs - next_action_probs).sum()
            loss = actor_loss + 0.5 * critic_loss + penalty_coeff * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print("Episode: {}, total reward: {}".format(episode, total_reward))

    env.close()


if __name__ == "__main__":
    ppo_penalty("CartPole-v1")