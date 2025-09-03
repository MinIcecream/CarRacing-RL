import numpy as np
import matplotlib.pyplot as plt
import glob
import os

log_dir = "../logs"

log_files = glob.glob(os.path.join(log_dir, "**/evaluations.npz"), recursive=True)

if not log_files:
    raise FileNotFoundError(f"No evaluations.npz found in {log_dir}")

plt.figure(figsize=(10, 6))

for log_file in log_files:
    data = np.load(log_file)
    timesteps = data["timesteps"]
    rewards = data["results"].mean(axis=1)  # Mean reward per eval

    plt.plot(timesteps, rewards, alpha=0.3, label=f"{os.path.basename(log_file)}")

plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Evaluation Performance Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
