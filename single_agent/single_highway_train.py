import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, DDPG
import torch

# Check the cuda configuration
print(torch.cuda.is_available())

# Initialize the environment for training
env = gym.make("highway-fast-v0", render_mode='human')
env.config['lanes_count'] = 4  # Set the lane to 4

# Initialize the DQN model for training
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=64,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="COMP0124_dqn/highway")
model.learn(int(2e4))
model.save("COMP0124_dqn/model_highway")