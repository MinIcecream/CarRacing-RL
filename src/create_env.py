from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TimeLimit
from wrappers.ensure_channel_wrapper import EnsureChannelWrapper
from wrappers.crop_wrapper import CropObservationWrapper
from wrappers.reward_clip_wrapper import RewardClipWrapper
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

def create_env(render_mode=None, n_envs = 4):
    def wrap_env(env):
        env = GrayscaleObservation(env, keep_dim=True)
        env = CropObservationWrapper(env, crop_bottom=12)    
        env = ResizeObservation(env, shape=(48, 48))        
        env = EnsureChannelWrapper(env)
        env = RewardClipWrapper(env, max_reward=1.0) 

        return env

    env = make_vec_env(
        "CarRacing-v3",
        n_envs=n_envs,
        wrapper_class=wrap_env,
        env_kwargs={"domain_randomize": True, "render_mode": render_mode, "continuous": False},
    )

    env = VecTransposeImage(env)

    env = VecFrameStack(env, n_stack=n_envs)

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return env