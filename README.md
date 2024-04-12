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

1. Download `highway_env` source code (https://github.com/Farama-Foundation/HighwayEnv.git) and `Heterogeneous_Highway_Env` source code (https://github.com/wuxiyang1996/Heterogeneous_Highway_Env.git) from github

2. Copy the `env`, `road` and `vehicle` folders from `heterogeneous_highway_env` to `highway_env/highway_env`, and overwrite any conflicting files.

3. copy the `util.py` file from `heterogeneous_highway_env` to `highway_env/highway_env` and overwrite the original `util.py` file.

4. Go to `highway_env` and run `pip install -e . `

5. `pip install stable-baselines3==1.8.0`.

6. `pip install gym==0.21.0`.

7. Run `hetero_highway.py` as a test environment, if it works, the environment is configured successfully!

# Progress

1. We have used Deep Q-Network to train monozygotes, the relevant code implementation can be found in the DQN_train folder.

2. We've re-edited the environment configuration file for intersection_env.py with the functions in it. It can be replaced intersection_env.py with the source code. Run hetero_highway_train.py and hetero_intersection_train.py will use multi-agent to train model.

# Reference

1. HighwayEnv, https://github.com/Farama-Foundation/HighwayEnv.git

2. Heterogeneous_Highway_Env, https://github.com/wuxiyang1996/Heterogeneous_Highway_Env.git

3. highway-env Documentation, https://highway-env.farama.org/installation/