import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, DDPG
import torch

# Check the cuda configuration
print(torch.cuda.is_available())

# Initialize the environment for training
env_modified = gym.make("highway-fast-v0", render_mode='human')
env_modified.config['lanes_count'] = 2
env_modified.config['vehicles_counts'] = 100

# Initialize the DQN model for training
model = DQN('MlpPolicy', env_modified,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="COMP0124_dqn/highway_modified")
model.learn(int(2e4))
model.save("COMP0124_dqn/model_highway_modified")