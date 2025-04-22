'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import gymnasium as gym
import numpy as np
import torch
from ppo_mpi_base.ppo import PPO

'''
run me with: mpiexec -n <number_of_procs> python train_mpi.py

'''


# generic network that we can use
class Network(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.fc1 = torch.nn.Linear(input_size, 256)
		self.fc2 = torch.nn.Linear(256, 256)
		self.fc3 = torch.nn.Linear(256, output_size)

	def forward(self, x):
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# define our environment
def env_fn(**kwargs):
	env = gym.make("Ant-v4")
	return env

# define a policy network architecture
def policy_net_fn(obs, act, **kwargs):
	return Network(obs, act)

# define a value network architecture
def value_net_fn(obs, **kwargs):
	return Network(obs, 1)


# run!
PPO(
    total_steps=10e6,

    env_fn=env_fn, 
    network_fn=policy_net_fn,
    value_network_fn=value_net_fn,

    seed=0, 
    rollout_length_per_worker=1024,  
    clip_rewards=False,
    gamma=0.99, 
    lam=0.97,
    max_ep_len=1000,
    clip_ratio=0.2,
    entropy_coef=0.01,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_batch_size=128,
    pi_grad_clip=1.0,
    v_grad_clip=1.0,
    log_directory="./output/logs/ant_mpi/", 
    save_location="./output/saved_model/ant_mpi/",
    device=None,

    train_pi_epochs=80, 
    train_v_epochs=80, 
    target_kl=0.01, 
)