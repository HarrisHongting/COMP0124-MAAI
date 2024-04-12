import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, DDPG
import torch
import matplotlib.pyplot as plt

# Initialize the highway environment for testing
test_env = gym.make("highway-v0", render_mode='human')

# Load and test saved model
model = DQN.load("COMP0124_dqn/model_highway")

# test the model
episode_rewards = []
num_episodes = 20

for episode in range(num_episodes):
  done = truncated = False
  obs, info = test_env.reset()
  total_rewards = 0
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    total_rewards += reward
    test_env.render()

  episode_rewards.append(total_rewards)


# Plot the reward
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Rewards')
plt.title('Performance Over Time')
plt.grid(True)
plt.show()