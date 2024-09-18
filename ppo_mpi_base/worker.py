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

import numpy as np
import torch
from torch.optim import Adam

try:
    import gymnasium as gym
except:
    import gym

import os
import time
import math
from torch.utils.tensorboard import SummaryWriter

from ppo_mpi_base.buffer import PPOBuffer
import ppo_mpi_base.core as core
from ppo_mpi_base.mpi_utils import *

class PPOWorker:

    def __init__(
        self, 
        env_fn,  
        network_fn,
        value_network_fn,
        seed=0, 
        rollout_length_per_worker=5000,  
        clip_rewards=False,
        gamma=0.99, 
        lam=0.95,
        max_ep_len=None,
        clip_ratio=0.2,
        entropy_coef=0.0,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_batch_size=None,
        pi_grad_clip=1.0,
        v_grad_clip=1.0,
        log_directory="./logs/", 
        save_location="./saved_model/",
        rollout_interception_callback=None
    ):

        if log_directory[-1] != "/": log_directory += "/"
        if save_location[-1] != "/": save_location += "/"

        if max_ep_len is None:
            max_ep_len = 1_000_000

        if train_batch_size is None:
            train_batch_size = rollout_length_per_worker

        self.rollout_length_per_worker = rollout_length_per_worker
        self.clip_rewards = clip_rewards
        self.gamma = gamma
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_batch_size = train_batch_size
        self.pi_grad_clip = pi_grad_clip
        self.v_grad_clip = v_grad_clip
        self.save_location = save_location
        self.best_performance = -100000.0
        self.last_rollout_global_perf = -100000.0
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.rollout_interception_callback = rollout_interception_callback

        # Random seed
        seed += 10000 * self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Logger
        if self.rank==0:
            self.summary_writer = SummaryWriter(log_dir=log_directory)

        # Instantiate environment
        self.env = env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape

        # Create actor-critic module
        self.ac = core.ActorCritic(self.env.observation_space, self.env.action_space, network_fn, value_network_fn)

        # Create buffer for storing rollout data
        self.buffer = PPOBuffer(obs_dim, act_dim, self.rollout_length_per_worker, gamma, lam)

        # Set up network optimizers
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        # Set up tracking of current episode
        try:
            self.env.seed(seed)
            self.curr_state = self.env.reset()
        except:
            self.curr_state = self.env.reset(seed=seed)
        self.curr_ep_len = 0
        self.curr_ep_ret = 0.0
        self.total_steps = 0

    # ==================================================================
    # Data collection and logging

    def rollout(self):
        # collect data for this loop
        self.buffer.reset_for_collection()
        self.local_epoch_results = []
        t = 0
        rolling = True
        curr_ep_tuples = []
        while rolling:

            a, v, logp = self.ac.step(torch.as_tensor(np.array(self.curr_state), dtype=torch.float32))
            next_o, r, d, info = self.env.step(a)
            self.curr_ep_len += 1
            self.curr_ep_ret += r

            # clip rewards (after we have stored the true value)
            if self.clip_rewards:
                if r>0.0:
                    r = 1.0
                elif r<0.0:
                    r = -1.0
                else:
                    r=0.0

            curr_ep_tuples.append((self.curr_state, a, r, v, logp))
            self.curr_state = next_o

            if self.max_ep_len is not None:
                timeout = self.curr_ep_len == self.max_ep_len
            else:
                timeout = False
            terminal = d or timeout

            last_step = (t==self.rollout_length_per_worker-1)

            if terminal or last_step:

                # should we consider the timeout to be terminal???????
                # if didn't hit episode end, bootstrap value target
                if timeout or last_step:
                    _, v, _ = self.ac.step(torch.as_tensor(np.array(self.curr_state), dtype=torch.float32))
                else:
                    v = 0

                # pass rollout through external hook, if any
                if self.rollout_interception_callback is not None:
                    curr_ep_tuples = self.rollout_interception_callback(curr_ep_tuples)

                # save to buffer
                for tup in curr_ep_tuples:
                    self.buffer.store(*tup)
                curr_ep_tuples = []

                # finish path (advantage calc)
                self.buffer.finish_path(v)

                if terminal:
                    # store (result, length, [when result was acquired])
                    self.local_epoch_results.append((self.curr_ep_ret, self.curr_ep_len, float(t)/self.rollout_length_per_worker))

                    # only reset if terminal?
                    self.curr_state = self.env.reset()
                    self.curr_ep_len = 0
                    self.curr_ep_ret = 0.0

            # update loop info
            t += 1

            if t>= self.rollout_length_per_worker:
                rolling = False


    def log(self):
        # collect results across all workers and write to logs
        all_results = mpi_gather_objects(MPI.COMM_WORLD, self.local_epoch_results)
        all_results = sum(all_results, [])
        rets = [x[0] for x in all_results]
        lens = [x[1] for x in all_results]
        zero_print(rets)

        if self.rank==0 and len(rets)>0:
            dsteps = MPI.COMM_WORLD.Get_size()*self.rollout_length_per_worker

            # not exact but it looks nice, and over millions of steps doesn't matter much...
            evenly_distributed_points = np.linspace(self.total_steps, self.total_steps+dsteps, len(rets)+1)
            for i in range(len(rets)):
                self.summary_writer.add_scalar("reward", rets[i], evenly_distributed_points[i+1])
                self.summary_writer.add_scalar("length", lens[i], evenly_distributed_points[i+1])
            self.summary_writer.flush()
        
        if len(rets)>0:
            self.last_rollout_global_perf = np.mean(rets)
        else:
            self.last_rollout_global_perf = -10000000

        self.total_steps += MPI.COMM_WORLD.Get_size()*self.rollout_length_per_worker
        if(len(rets)>0):
            zero_print("Average Return:", round(np.mean(rets),3), "Total Steps:", f'{self.total_steps:,}')
        else:
            zero_print("Average Return: - Total Steps:", f'{self.total_steps:,}')


    def save(self):
        if MPI.COMM_WORLD.Get_rank()==0:

            # only save if new best performance
            best = False
            if self.last_rollout_global_perf >= self.best_performance:
                self.best_performance = self.last_rollout_global_perf
                best = True

            # save
            if self.save_location is not None:
                if not os.path.exists(self.save_location):
                    os.makedirs(self.save_location)
                torch.save(self.ac.state_dict(), self.save_location+"model_latest.pt")

                if best:
                    torch.save(self.ac.state_dict(), self.save_location+"model_best.pt")



    # ==================================================================
    # Routines to compute and apply updates to the policy network or combined network

    def get_and_shuffle_data(self):
        data = self.buffer.get()
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']
        perm = np.random.permutation(len(obs))
        self.epoch_data = {
            "obs":obs[perm],
            "act":act[perm],
            "adv":adv[perm],
            "logp":logp_old[perm],
            "ret":ret[perm],
        }
        self.batch = 0

    def compute_grads_for_policy_network(self):
        self.pi_optimizer.zero_grad()
        data = self.epoch_data
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']
        obs = obs[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        act = act[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        adv = adv[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        logp_old = logp_old[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        ret = ret[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]

        pi, logp, entropy = self.ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        loss_pi -= self.entropy_coef*torch.mean(entropy)
        loss_pi.backward()

        approx_kl = (logp_old - logp).mean().item()

        self.batch += 1
        return approx_kl

    def average_grads_for_policy_network(self):
        average_grads_across_processes(MPI.COMM_WORLD, self.ac.pi.parameters())

    def update_policy_network(self):
        if self.pi_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.pi_grad_clip)
        self.pi_optimizer.step()

    def sync_weights_for_policy_network(self):
        sync_weights_across_processes(MPI.COMM_WORLD, self.ac.pi.parameters())

    # ==================================================================
    # Routines to compute and apply updates to the value network

    def compute_grads_for_value_network(self):
        self.vf_optimizer.zero_grad()
        data = self.epoch_data
        obs, ret = data['obs'], data['ret']
        obs = obs[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        ret = ret[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        v_pred = self.ac.v(obs)
        loss = ((v_pred - ret)**2).mean()
        loss.backward()
        self.batch += 1

    def average_grads_for_value_network(self):
        average_grads_across_processes(MPI.COMM_WORLD, self.ac.v.parameters())

    def update_value_network(self):
        if self.v_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.v_grad_clip)
        self.vf_optimizer.step()

    def sync_weights_for_value_network(self):
        sync_weights_across_processes(MPI.COMM_WORLD, self.ac.v.parameters())


