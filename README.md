# PPO MPI Base

This is a base version of PPO that is meant as a starting point for algorithm modifications or as a simple implementation to use in research. It is refactored from spinning-up and includes some additional functionality here and there.

This work is authored by Ted Staley and is Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC, please see the LICENSE file.



## Installation

Requires torch, tensorboard, mpi4py, numpy, scipy, and gym(nasium).

Most of these can be pip-installed. To install mpi4py I recommend conda-forge:
```
conda install -c conda-forge mpi4py
```

Then install this repo with:
```
pip install -e .
```



## Example Usage

See ```train.py``` in each of the examples. In general, you need to define three methods and pass these to ppo_mpi_base.ppo.PPO():

- A method that builds and returns an environment instance, applying all wrappers, etc
- A method that builds and returns a policy network
- A method that builds and returns a value network

Together these give a great deal of flexibility over the experiment (along with many hyperparameters you can set). You then run this file using MPI:

```mpiexec -n 8 python train.py```



## Changes from Spinning-Up

- Refactored to separate the high-level algorithm (ppo.py) from the lower-level details (worker.py). I find this useful if I want to hack at the algorithm. It is also easier to "see" the big picture this way. But there are more files now.
- Can pass in arbitrary networks for policy and value function (i.e. CNNs)
- Added entropy loss option
- Supports gymansium
- Decoupled rollout length from max episode length
- Decoupled rollout length from batch size
- No longer supports coupled policy-value networks
- A few minor things: 
  - Logging to tensorboard
  - Model saving triggered by new performance best
  - Estimated FPS and time remaining printouts



## Results on Selected Environments

### Atari
**examples/atari/**

Requires atari: ```pip install gymnasium[atari], pip install gymnasium[accept-rom-license]```. These environments proved quite challenging to solve as there are lots of details in the wrapping of the environment and hyperparameters. The current settings work alright but do not match PPO's reported results. These ran for 20M steps in an attempt to reach better performance.

Additionally, these runs use the "NoFrameskip-v4" environments over the "ALE...v5" environments.

![atari](./assets/atari.png)

For comparison, the original PPO paper reports scores of 274.8, 20.7, 1204.5, and 942.5 for Breakout, Pong, Seaquest, and Space Invaders, respectively.

### Mujoco

**examples/mujoco/**

Requires mujoco: ```pip install gymnasium[mujoco]```. These are much more stable than the Atari environments above. However, if attempting these types of problems SAC is probably much more performant.

![mujoco](./assets/mujoco.png)

For comparison, the original PPO paper reports scores of around 1800, 2250, and 3000 for HalfCheetah, Hopper, and Walker2d respectively. They used the v1 environments and did not include Ant.

