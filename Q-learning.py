import numpy as np

# SARSA算法是同策略算法，而Q-learning是异策略算法。
# 同策略算法指的是在更新动作值函数时，使用当前策略选择的动作来更新，
# 即使用SARSA算法中的next_action来更新动作值函数。
# 而异策略算法指的是在更新动作值函数时，使用贪心策略选择的动作来更新，
# 即使用Q-learning中的argmax(Q[next_state])来更新动作值函数。
# ref: https://blog.csdn.net/v_JULY_v/article/details/128965854?login=from_csdn

# 定义迷宫环境
class MazeEnv:
    def __init__(self):
        self.grid = np.zeros((4, 4))  # 迷宫网格
        self.grid[0, 0] = 1  # 起始位置
        self.grid[3, 3] = 2  # 目标位置
        self.state = (0, 0)  # 初始状态
        self.done = False  # 是否到达目标位置

    def step(self, action):
        x, y = self.state
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1

        x = np.clip(x, 0, 3)  # 确保不超出边界
        y = np.clip(y, 0, 3)

        self.state = (x, y)
        self.done = (self.grid[x, y] == 2)  # 判断是否到达目标位置

        if self.done:
            reward = 1  # 到达目标位置的奖励
        else:
            reward = 0  # 其他情况的奖励

        return self.state, reward, self.done

    def reset(self):
        self.state = (0, 0)
        self.done = False
        return self.state

# 定义Q-learning代理
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))  # 动作值函数表
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子γ

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:  # ε-greedy策略
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q(self, state, action, reward, next_state):
        max_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state, max_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

# 创建迷宫环境和代理
env = MazeEnv()
agent = QLearningAgent(num_states=16, num_actions=4, learning_rate=0.1, discount_factor=0.9)

# 进行Q-learning
num_episodes = 10
epsilon = 0.1  # ε-greedy策略中的ε
for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state, epsilon)
        next_state, reward, done = env.step(action)
        agent.update_Q(state, action, reward, next_state)
        state = next_state

# 输出学习到的动作值函数
print(agent.Q)