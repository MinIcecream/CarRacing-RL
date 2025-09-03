from create_env import create_env
from stable_baselines3 import PPO

def get_model():
    train_env = create_env()

    def linear_schedule(initial_value, final_value=0.0):
        def func(progress_remaining: float):
            return final_value + (initial_value - final_value) * progress_remaining
        return func
    
    policy_kwargs = dict(
        normalize_images=False  # tell the CNN that images are already normalized
    )

    return PPO(
        "CnnPolicy",    # CNN for image input
        train_env,
        verbose=1,
        ent_coef=0.01,
        learning_rate=linear_schedule(3e-4, 5e-5),
        n_steps=4096,
        batch_size=512,
        policy_kwargs=policy_kwargs,
    ) 