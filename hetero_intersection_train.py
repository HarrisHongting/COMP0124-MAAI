# Intersection scenario trained and run

import pprint
import random
from matplotlib import pyplot as plt
from PIL import Image

import gym
import __editable___highway_env_1_4_finder as highway_env
from stable_baselines3 import DQN

from gym.envs.registration import register

register(
    id='intersection-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentIntersectionEnv',
)

env = gym.make("intersection-multi-agent-v0")
pprint.pprint(env.config)
env.config['controlled_vehicles'] = 3
obs = env.reset()


model = DQN('MlpPolicy', env,
             policy_kwargs=dict(net_arch=[256, 256]),
             learning_rate=5e-4,
             buffer_size=15000,
             learning_starts=200,
             batch_size=2048,
             gamma=0.8,
             train_freq=1,
             gradient_steps=1,
             target_update_interval=50,
             verbose=1,
             tensorboard_log="intersection_dqn/")
model.learn(int(2e4))
model.save("intersection_dqn/model")

model = DQN.load("intersection_dqn/model")


while True:
    done = truncated = False
    obs = env.reset()
    epoch = 1
    id = 1
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        images = env.render(mode='rgb_array')
        img = Image.fromarray(images)
        os.mkdir(f'./intersection_images_{epoch}')
        img.save(f'./intersection_images_{epoch}/'+str(id)+'.jpg')
        id += 1

    epoch += 1