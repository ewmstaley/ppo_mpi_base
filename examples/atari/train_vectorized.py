'''
Copyright © 2025 The Johns Hopkins University Applied Physics Laboratory LLC
 
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
import ale_py
import numpy as np
import torch
from ppo_mpi_base.ppo import PPO

'''
run me without mpi: python train_vectorized.py

'''

# generic CNN that we can use
class CNNNetwork(torch.nn.Module):
    def __init__(self, output_size, final_gain=1.0):
        super().__init__()

        # from ray's setup
        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.fc1 = torch.nn.Linear(128*9, 512)
        self.fc2 = torch.nn.Linear(512, output_size)

        torch.nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv4.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc2.weight, gain=final_gain)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.flatten(x, 1) if len(x.shape)==4 else torch.flatten(x, 0)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# define our environment (a single environment instance)
# note no kwargs here
def env_fn():

    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
        env, 
        terminal_on_life_loss=False, # recommended by sb3
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )

    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


# define how to construct a vectorized environment
def make_vec_env(env_fn, num_envs):
    env = gym.vector.AsyncVectorEnv([env_fn]*num_envs)
    return env


# define a policy network architecture
def policy_net_fn(obs, act, **kwargs):
    return CNNNetwork(act, 0.01)


# define a value network architecture
def value_net_fn(obs, **kwargs):
    return CNNNetwork(1, 1.0)


if __name__ == "__main__":

    device = torch.device("cuda")

    # run!
    # Note our batch size needs to be N times as large (if we are trying to mimic mpi version)
    # Note we set a device for the networks and some extra settings for vectorization
    PPO(
        total_steps=20e6,

        env_fn=env_fn, 
        network_fn=policy_net_fn,
        value_network_fn=value_net_fn,

        seed=0, 
        rollout_length_per_worker=512, # (per environment)
        clip_rewards=False,
        gamma=0.99, 
        lam=0.95,
        max_ep_len=None,
        clip_ratio=0.1,
        entropy_coef=0.01,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_batch_size=32 * 8,
        pi_grad_clip=1.0,
        v_grad_clip=1.0,
        log_directory="./output/logs/pong_vect/", 
        save_location="./output/saved_model/pong_vect/",
        device=device,

        train_pi_epochs=5, 
        train_v_epochs=5, 
        target_kl=0.01, 

        # vector settings
        use_vectorized_envs=True,
        num_environments=8,
        make_vec_env_fn=make_vec_env
    )