import gymnasium as gym
import numpy as np

class RewardClipWrapper(gym.Wrapper):
    def __init__(self, env, max_reward=1.0):
        super().__init__(env)
        self.max_reward = max_reward

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        reward = np.clip(reward, a_min=None, a_max=self.max_reward)
        return obs, reward, done, truncated, info
