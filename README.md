# CarRacing-PPO

An agent trained to play Gymnasium's CarRacing environment. The agent was trained using Proximal Policy Optimisation (PPO), and a CNN was used as the neural network. The agent was trained using StableBaselines3 over 5,000,000 timesteps, reaching a mean reward of 832.

### Strategies
Several strategies were used to make the model perform better, including:
- Faster training using 4 vectorized environments
- Used 4-frame stacking for temporal context
- Discretized the action space
- Cropped the frame, made it grayscale, and reduced it to 48x48 for faster training
- Normalized the observations for better performance
- Clipped rewards to reduce large swings

### Demo of the final trained model playing:
![car_racing](https://github.com/user-attachments/assets/98d627a1-4135-44df-9dda-ae1e97962afa)

### Mean Reward graph:
<img width="700" height="450" alt="image" src="https://github.com/user-attachments/assets/181c31d1-790e-4021-9f82-1319b3c72898" />
