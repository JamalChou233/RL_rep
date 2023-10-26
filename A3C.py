import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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


# 定义worker函数
def worker(worker_id, global_policy, num_epochs, num_steps, gamma, lr):
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    local_policy = Policy(input_dim, output_dim)
    local_policy.load_state_dict(global_policy.state_dict())
    optimizer = optim.Adam(local_policy.parameters(), lr=lr)

    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(num_steps):
            state = torch.FloatTensor(state)
            action_probs, value = local_policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward

            next_state = torch.FloatTensor(next_state)
            _, next_value = local_policy(next_state)

            if done:
                target_value = torch.FloatTensor([reward])
            else:
                target_value = reward + gamma * next_value

            advantage = target_value - value

            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage
            critic_loss = advantage.pow(2)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            state = next_state.numpy()

            if done:
                break

        print(f"Worker: {worker_id}, Epoch: {epoch}, Total reward: {total_reward}")

    env.close()


# 定义A3C算法
def a3c(num_processes, num_epochs, num_steps, gamma, lr):
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    global_policy = Policy(input_dim, output_dim)
    global_policy.share_memory()
    optimizer = optim.Adam(global_policy.parameters(), lr=lr)

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(i, global_policy, num_epochs, num_steps, gamma, lr))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# 运行A3C算法
a3c(num_processes=4, num_epochs=100, num_steps=200, gamma=0.99, lr=0.001)