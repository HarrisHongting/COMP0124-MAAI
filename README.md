# COMP0124-MAAI
This is all related code of our COMP0124 coursework 3: Multi-agent reinforcement learning in the highway scenario

# Main environmental requirements

- python3.8

- pytorch

- highway_env==1.4.0

- Heterogeneous_Highway_Env

- stable_baselines3==1.8.0

- gym==0.21.0

# Installation process

1. Download `highway_env` source code and `Heterogeneous_Highway_Env` source code from github

2. Copy the `env`, `road` and `vehicle` folders from `heterogeneous_highway_env` to `highway_env/highway_env`, and overwrite any conflicting files.

3. copy the `util.py` file from `heterogeneous_highway_env` to `highway_env/highway_env` and overwrite the original `util.py` file.

4. Go to `highway_env` and run `pip install -e . `

5. `pip install stable-baselines3==1.8.0`.

6. `pip install gym==0.21.0`.

7. Run `hetero_highway.py` in the appendix, if it works, the environment is configured successfully!

# Appendix

## hetero_highway.py（As a test environment）

```python
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

```

## hetero_highway_train.py(Highway scenario trained and run)

```python
import os
import pprint
import random
from matplotlib import pyplot as plt
from PIL import Image


import gym
import __editable___highway_env_1_4_finder as highway_env
from stable_baselines3 import DQN

from gym.envs.registration import register

register(
    id='highway-hetero-v0',
    entry_point='highway_env.envs:HighwayEnvHetero',
)

env = gym.make("highway-hetero-v0")
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
             tensorboard_log="highway_dqn/")
model.learn(int(1e3))    # Number of study rounds
model.save("highway_dqn/model")    # Model parameter save location

model = DQN.load("highway_dqn/model")    # load modle


while True:
    done = truncated = False
    obs = env.reset()
    epoch = 1
    id = 1
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        images = env.render()
        img = Image.fromarray(images)
        os.mkdir(f'./highway_images_{epoch}')
        img.save(f'./highway_images_{epoch}/'+str(id)+'.jpg') 
        id += 1

    epoch += 1
```



## hetero_intersection_train.py(Intersection scenario trained and run)

```python
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


```