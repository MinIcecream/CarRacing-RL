import gymnasium as gym
import numpy as np

class CropObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_bottom=12):
        super().__init__(env)
        self.crop_bottom = crop_bottom

        obs_shape = self.observation_space.shape
        # Adjust the height of the observation space
        new_height = obs_shape[0] - crop_bottom
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(new_height, obs_shape[1], obs_shape[2]),
            dtype=np.uint8
        )

    def observation(self, obs):
        # Crop the bottom N rows
        return obs[:-self.crop_bottom, :, :]
