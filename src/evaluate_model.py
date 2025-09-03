
from create_env import create_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO

eval_env = create_env("rgb_array")

video_folder = "../videos/"
eval_env = VecVideoRecorder(
    eval_env,
    video_folder,
    record_video_trigger=lambda x: True,  # record every episode
    video_length=2000,                    # max timesteps to record
    name_prefix="carracing_run"
)

model = PPO.load("../logs/best_model")

obs = eval_env.reset()
done = [False]

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

eval_env.close()

print(f"Video saved to {video_folder}")
