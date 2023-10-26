import numpy as np

# 定义迷宫环境
class MazeEnv:
    def __init__(self):
        self.num_states = 25
        self.num_actions = 4
        self.rewards = np.zeros((self.num_states, self.num_actions))
        self.rewards[0, 1] = -1
        self.rewards[1, 0] = -1
        self.rewards[1, 2] = -1
        self.rewards[2, 1] = -1
        self.rewards[2, 3] = -1
        self.rewards[3, 2] = -1
        self.rewards[3, 4] = -1
        self.rewards[4, 3] = -1
        self.rewards[5, 1] = -1
        self.rewards[5, 5] = -1
        self.rewards[6, 2] = -1
        self.rewards[6, 6] = -1
        self.rewards[7, 3] = -1
        self.rewards[7, 8] = -1
        self.rewards[8, 4] = -1
        self.rewards[8, 7] = -1
        self.rewards[8, 9] = -1
        self.rewards[9, 5] = -1
        self.rewards[9, 8] = -1
        self.rewards[9, 10] = -1
        self.rewards[10, 6] = -1
        self.rewards[10, 9] = -1
        self.rewards[10, 11] = -1
        self.rewards[11, 7] = -1
        self.rewards[11, 10] = -1
        self.rewards[11, 12] = -1
        self.rewards[12, 11] = -1
        self.rewards[12, 13] = -1
        self.rewards[13, 12] = -1
        self.rewards[13, 14] = -1
        self.rewards[14, 13] = -1
        self.rewards[14, 15] = -1
        self.rewards[15, 14] = -1
        self.rewards[15, 16] = -1
        self.rewards[16, 15] = -1
        self.rewards[16, 17] = -1
        self.rewards[17, 16] = -1
        self.rewards[17, 18] = -1
        self.rewards[18, 17] = -1
        self.rewards[18, 19] = -1
        self.rewards[19, 18] = -1
        self.rewards[19, 20] = -1
        self.rewards[20, 19] = -1
        self.rewards[20, 21] = -1
        self.rewards[21, 20] = -1
        self.rewards[21, 22] = -1
        self.rewards[22, 21] = -1
        self.rewards[22, 23] = -1
        self.rewards[23, 22] = -1
        self.rewards[23, 24] = -1
        self.rewards[24, 23] = 1

    def reset(self):
        return 0

    def step(self, state, action):
        if state == 24:  # 终止状态
            return state, 1, True
        else:
            if action == 0:  # 上
                next_state = state - 5 if state >= 5 else state
            elif action == 1:  # 下
                next_state = state + 5 if state < 20 else state
            elif action == 2:  # 左
                next_state = state - 1 if state % 5 != 0 else state
            elif action == 3:  # 右
                next_state = state + 1 if state % 5 != 4 else state
            reward = self.rewards[state, action]
            done = False
            if next_state == 24:  # 终止状态
                done = True
            return next_state, reward, done

# 定义TD代理
class TDAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))  # 动作值函数表
        self.learning_rate = learning_rate  # 学习率α
        self.discount_factor = discount_factor  # 折扣因子γ

    def td_learning(self, env, num_episodes):
        for _ in range(num_episodes):
            state = env.reset()
            while True:
                action = np.argmax(self.Q[state, :])  # 根据动作值函数选择动作
                next_state, reward, done = env.step(state, action)
                td_target = reward + self.discount_factor * np.max(self.Q[next_state, :])  # TD目标
                td_error = td_target - self.Q[state, action]  # TD误差
                self.Q[state, action] += self.learning_rate * td_error  # 更新动作值函数表
                state = next_state
                if done:
                    break

# 创建迷宫环境和代理
env = MazeEnv()
agent = TDAgent(num_states=25, num_actions=4, learning_rate=0.1, discount_factor=0.9)

# 进行TD学习
agent.td_learning(env, num_episodes=1000)

# 输出动作值函数表
print(agent.Q)