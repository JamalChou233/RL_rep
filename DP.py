import numpy as np

# 定义网格世界环境
class GridWorldEnv:
    def __init__(self):
        self.num_states = 16
        self.num_actions = 4
        self.grid = np.zeros((4, 4))  # 网格世界
        self.grid[0, 3] = 1  # 终止状态
        self.rewards = np.zeros((self.num_states, self.num_actions))  # 奖励函数
        self.rewards[0, 1] = -1
        self.rewards[0, 2] = -1
        self.rewards[1, 0] = -1
        self.rewards[1, 2] = -1
        self.rewards[2, 0] = -1
        self.rewards[2, 3] = -1
        self.rewards[3, 0] = -1
        self.rewards[3, 1] = -1
        self.rewards[3, 3] = 1

    def reset(self):
        return 0

    def step(self, state, action):
        if state == 0 or state == 15:  # 终止状态
            return state, 0, True
        else:
            if action == 0:  # 上
                next_state = state - 4 if state >= 4 else state
            elif action == 1:  # 下
                next_state = state + 4 if state < 12 else state
            elif action == 2:  # 左
                next_state = state - 1 if state % 4 != 0 else state
            elif action == 3:  # 右
                next_state = state + 1 if state % 4 != 3 else state
            reward = self.rewards[state, action]
            done = False
            if next_state == 0 or next_state == 15:  # 终止状态
                done = True
            return next_state, reward, done

# 定义值迭代代理
class ValueIterationAgent:
    def __init__(self, num_states, num_actions, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.V = np.zeros(num_states)  # 状态值函数表
        self.P = np.zeros((num_states, num_actions))  # 策略表
        self.discount_factor = discount_factor  # 折扣因子γ

    def value_iteration(self, env):
        epsilon = 1e-6  # 收敛阈值
        while True:
            delta = 0
            for state in range(self.num_states):
                v = self.V[state]
                max_value = float('-inf')
                for action in range(self.num_actions):
                    next_state, reward, _ = env.step(state, action)
                    value = reward + self.discount_factor * self.V[next_state]
                    if value > max_value:
                        max_value = value
                self.V[state] = max_value
                delta = max(delta, abs(v - self.V[state]))
            if delta < epsilon:
                break

    def extract_policy(self, env):
        for state in range(self.num_states):
            max_value = float('-inf')
            best_action = 0
            for action in range(self.num_actions):
                next_state, reward, _ = env.step(state, action)
                value = reward + self.discount_factor * self.V[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            self.P[state, best_action] = 1

# 创建网格世界环境和代理
env = GridWorldEnv()
agent = ValueIterationAgent(num_states=16, num_actions=4, discount_factor=0.9)

# 进行值迭代
agent.value_iteration(env)

# 提取最优策略
agent.extract_policy(env)

# 输出最优策略
print(agent.P)