import pprint
import random
from matplotlib import pyplot as plt


import gym
import __editable___highway_env_1_4_finder as highway_env
frm gym.envs.registration import register

register(
    id='highway-hetero-v0',
    entry_point='highway_env.envs:HighwayEnvHetero',
)

env = gym.make("highway-hetero-v0")
pprint.pprint(env.config)
env.config['controlled_vehicles'] = 3 # change the number of vehicles
obs = env.reset()


pprint.pprint(env.config)    # Compare the difference between the two configs, other information in the config can also be modified

for _ in range(300):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    images = env.render(mode='rgb_array')
    plt.imshow(highway_images)
    plt.show()