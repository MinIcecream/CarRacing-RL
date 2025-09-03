from create_env import create_env
from model import get_model
from stable_baselines3.common.callbacks import EvalCallback

eval_env = create_env()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="../logs/",
    log_path="../logs/",
    eval_freq=25_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

model = get_model()
    
model.learn(total_timesteps=5_000_000, progress_bar=True, callback=eval_callback)