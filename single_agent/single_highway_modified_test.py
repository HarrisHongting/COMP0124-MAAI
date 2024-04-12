import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, DDPG
import torch
import matplotlib.pyplot as plt

# Initialize the highway environment for testing
test_env_modified = gym.make("highway-v0", render_mode='human')
test_env_modified.config['lanes_count'] = 2
test_env_modified.config['vehicles_counts'] = 100

# Load and test saved model
model = DQN.load("COMP0124_dqn/model_highway_modified")

# test the model
episode_rewards_modified = []
num_episodes = 20

for episode in range(num_episodes):
  done = truncated = False
  obs, info = test_env_modified.reset()
  total_rewards_modified = 0
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env_modified.step(action)
    total_rewards_modified += reward
    test_env_modified.render()

  episode_rewards_modified.append(total_rewards_modified)


# Plot two curves
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards_modified, label = "modified") # Reward for 2 lanes and 100 vehicles
plt.xlabel('Episode')
plt.ylabel('Total Rewards')
plt.title('Performance Over Time')
plt.grid(True)
plt.show()