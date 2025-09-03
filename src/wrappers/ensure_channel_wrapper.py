import gymnasium as gym
import numpy as np

class EnsureChannelWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=-1)
        return obs
