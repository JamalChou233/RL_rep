import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


def compute_advantages(rewards, values, gamma=0.99, lam=0.97):
    deltas = []
    advantages = []
    prev_value = 0
    prev_advantage = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * prev_value - v
        prev_value = v
        deltas.append(delta)

    deltas = list(reversed(deltas))

    for delta in deltas:
        advantage = delta + gamma * lam * prev_advantage
        prev_advantage = advantage
        advantages.append(advantage)

    advantages = list(reversed(advantages))

    return advantages


def train(env_name, lr=0.001, gamma=0.99, lam=0.97, max_episodes=1000, max_steps=1000):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    print(f"input_dim: {input_dim}")
    print(f"output_dim: {output_dim}")

    model = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_list = []
    reward_list = []

    for episode in range(max_episodes):
        state = env.reset()
        state = state[0]
        # print(f"state: {state}")
        done = False
        rewards = []
        values = []
        log_probs = []

        while not done:
            # action_probs, value = model(torch.FloatTensor(state))
            action_probs, value = model.forward(torch.FloatTensor(state))
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # print(f"dist: {dist}")
            # print(f"action: {action}")
            # print(f"log_prob: {log_prob}")
            # print(f"env.step(action.item()): ", env.step(action.item()))
            next_state, reward, done, _, info = env.step(action.item())

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state

            if len(rewards) >= max_steps:
                break

        if done:
            next_value = 0
        else:
            # next_value = model(torch.FloatTensor(next_state))[1].item()
            next_value = model.forward(torch.FloatTensor(next_state))[1].item()

        advantages = compute_advantages(rewards + [next_value], values)

        returns = []
        tmp = rewards + [next_value]
        sum_ = 0
        for i in range(len(tmp)-1, -1, -1):
            sum_ += tmp[i]
            returns.insert(0, sum_)
        # returns = sum(rewards)

        actor_loss = 0
        critic_loss = 0

        for log_prob, value, advantage, return_ in zip(log_probs, values, advantages, returns):
            actor_loss -= log_prob * advantage
            critic_loss += (value - return_) ** 2

        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer.step()

        # returns_sum = sum(returns[])
        print("Episode: {}, total reward: {}".format(episode, returns[0]))
        episode_list.append(episode)
        reward_list.append(returns[0])

    env.close()

    fig = plt.figure()
    plt.plot(episode_list, reward_list)
    plt.show()

if __name__ == "__main__":
    train("CartPole-v1", max_episodes=100)