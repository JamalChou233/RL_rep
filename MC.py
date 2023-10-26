import numpy as np

# 定义赌博游戏环境
class GambleEnv:
    def __init__(self):
        self.state = 1  # 初始状态
        self.done = False  # 游戏是否结束

    def step(self, action):
        if action == 0:  # 投注全部金额
            if np.random.rand() < 0.4:  # 以0.4的概率赢得游戏
                reward = self.state
            else:
                reward = -self.state
            self.done = True
        else:  # 不投注
            reward = 0
            self.done = True

        return self.state, reward, self.done

    def reset(self):
        self.state = 1
        self.done = False
        return self.state

# 定义蒙特卡洛强化学习代理
class MonteCarloAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))  # 动作值函数表
        self.N = np.zeros((num_states, num_actions))  # 每个状态动作对的访问次数
        self.epsilon = 0.1  # ε-greedy策略中的ε

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # ε-greedy策略
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q(self, episode):
        states, actions, rewards = zip(*episode)
        discounts = np.array([0.9**i for i in range(len(rewards))])  # 折扣因子γ
        G = np.sum(rewards * discounts)  # 计算回报值
        for i, (state, action) in enumerate(zip(states, actions)):
            self.N[state, action] += 1
            alpha = 1 / self.N[state, action]  # 更新步长
            self.Q[state, action] += alpha * (G - self.Q[state, action])

# 创建赌博游戏环境和代理
env = GambleEnv()
agent = MonteCarloAgent(num_states=2, num_actions=2)

# 进行强化学习
num_episodes = 10000
for _ in range(num_episodes):
    episode = []
    state = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        print('i: ', i)
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    agent.update_Q(episode)

# 输出学习到的动作值函数
print(agent.Q)