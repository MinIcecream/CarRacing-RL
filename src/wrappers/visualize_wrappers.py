import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation
from ensure_channel_wrapper import EnsureChannelWrapper
from crop_wrapper import CropObservationWrapper
from reward_clip_wrapper import RewardClipWrapper
from timeout_wrapper import TimeoutWrapper
from gymnasium.wrappers import ResizeObservation

# -------------------------------
# Custom wrapper stack
# -------------------------------
def make_wrapped_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # 1. Convert to grayscale
    env = GrayscaleObservation(env, keep_dim=True)

    # 2. Timeout wrapper
    env = TimeoutWrapper(env, max_offtrack_steps=50)

    # 3. Reward clipping
    env = RewardClipWrapper(env, max_reward=1.0)

    # 4. Crop bottom pixels
    env = CropObservationWrapper(env, crop_bottom=12)

    # 5. Resize to 84x84 (keeps channels last)
    env = ResizeObservation(env, shape=(48, 48))

    env = EnsureChannelWrapper(env)

    return env

# -------------------------------
# Create environment
# -------------------------------
env = gym.make("CarRacing-v3", render_mode="rgb_array")
wrapped_env = make_wrapped_env()

# -------------------------------
# Reset environments
# -------------------------------
obs_orig, info_orig = env.reset()
obs_wrapped, info_wrapped = wrapped_env.reset()

# -------------------------------
# Step once for proper rendering
# -------------------------------
for _ in range(50):
    action = wrapped_env.action_space.sample()
    obs_orig, reward, done, truncated, info = env.step(action)
    obs_wrapped, reward, done, truncated, info = wrapped_env.step(action)

# -------------------------------
# Plot before and after
# -------------------------------
plt.figure(figsize=(10, 5))

# Original RGB frame
plt.subplot(1, 2, 1)
plt.imshow(obs_orig)
plt.title("Original Observation (RGB)")
plt.axis("off")

# Wrapped frame (grayscale + crop + resize)
plt.subplot(1, 2, 2)
if obs_wrapped.ndim == 3 and obs_wrapped.shape[-1] == 1:
    plt.imshow(obs_wrapped[:, :, 0], cmap="gray")
else:
    plt.imshow(obs_wrapped)
plt.title("After Wrappers (Grayscale + Crop + Resize)")
plt.axis("off")

plt.tight_layout()
plt.show()
