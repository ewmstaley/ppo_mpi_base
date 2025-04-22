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

import numpy as np
import torch
from torch.optim import Adam

import os
import time
import math

from ppo_mpi_base.worker_base import PPOWorkerBase
from ppo_mpi_base.buffer import PPOBuffer
import ppo_mpi_base.core as core
from ppo_mpi_base.mpi_utils import *

'''
Worker for decentralized PPO: each process has networks + env.
Between processes, gradients are shared.
'''

class PPOMPIWorker(PPOWorkerBase):

    def post_init(self):

        assert self.device is None, "Please pass device=None for mpi runs (cpu only)."

        # Random seed
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.world_size = MPI.COMM_WORLD.Get_size()
        seed = 10000 * self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env_seed = seed


    def build_env(self):
        self.env = self.env_fn(**self.env_kwargs)
        self.obs_space_reference = self.env.observation_space
        self.act_space_reference = self.env.action_space

        # Create buffer for storing rollout data
        self.buffer = PPOBuffer(
            self.obs_space_reference.shape, 
            self.act_space_reference.shape, 
            self.rollout_length_per_worker, 
            self.gamma, 
            self.lam
        )

        # Set up tracking of current episode
        try:
            self.env.seed(self.env_seed)
            self.curr_state = self.env.reset()
        except:
            self.curr_state = self.env.reset(seed=self.env_seed)

        # discard the initial info
        self.curr_state = self.curr_state[0]
        self.curr_state = self.state_processing_hook(self.curr_state)

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
            next_o, r, d1, d2, info = self.env.step(a)
            d = (d1 or d2)
            self.curr_ep_len += 1
            self.curr_ep_ret += r
            next_o = self.state_processing_hook(next_o)

            # clip rewards (after we have stored the true value)
            if self.clip_rewards:
                r = np.sign(r)

            curr_ep_tuples.append((self.curr_state, a, r, v, logp))
            self.curr_state = next_o

            if self.max_ep_len is not None:
                timeout = self.curr_ep_len == self.max_ep_len
            else:
                timeout = False
            terminal = d or timeout

            last_step = (t==self.rollout_length_per_worker-1)

            if terminal or last_step:

                # should we consider the timeout to be terminal?
                # if didn't hit episode end, bootstrap value target
                if (timeout or last_step) and not terminal:
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
                    self.curr_state, _ = self.env.reset()
                    self.curr_ep_len = 0
                    self.curr_ep_ret = 0.0

            # update loop info
            t += 1

            if t>= self.rollout_length_per_worker:
                rolling = False

    # ==================================================================
    # Logging and Saving

    def maybe_gather_results(self):
        return mpi_gather_objects(MPI.COMM_WORLD, self.local_epoch_results)

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

    def average_grads_for_policy_network(self):
        average_grads_across_processes(MPI.COMM_WORLD, self.ac.pi.parameters())

    def sync_weights_for_policy_network(self):
        sync_weights_across_processes(MPI.COMM_WORLD, self.ac.pi.parameters())

    # ==================================================================
    # Routines to compute and apply updates to the value network

    def average_grads_for_value_network(self):
        average_grads_across_processes(MPI.COMM_WORLD, self.ac.v.parameters())

    def sync_weights_for_value_network(self):
        sync_weights_across_processes(MPI.COMM_WORLD, self.ac.v.parameters())


