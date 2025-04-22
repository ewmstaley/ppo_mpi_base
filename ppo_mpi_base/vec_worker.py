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
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from ppo_mpi_base.worker_base import PPOWorkerBase
from ppo_mpi_base.buffer import PPOBuffer
import ppo_mpi_base.core as core

'''
Worker for centralized PPO: only one copy of each network, but many envs in parallel.
Only one of these will be created.
Between processes, rollout data is shared (instead of gradients).

This class maintains a separate ppo buffer for each environment instance, which
makes some of the rollout logic complicated. It is very similar to the mpi_worker,
but adds loops for iterating over the collection of environments. Use mpi_worker
as a reference.
'''
class PPOVecWorker(PPOWorkerBase):

    def post_init(self):
        self.rank = 0
        self.world_size = self.num_environments
        torch.manual_seed(self.original_seed)
        np.random.seed(self.original_seed)

        assert self.max_ep_len is None, "Vectorized setting does not support max ep len; add this with TimeLimit wrapper instead."

    def build_env(self):
        assert self.env_kwargs == {}, "Cannot use kwargs for vectorized envs; not supported by gymnasium."
        self.env = self.make_vec_env_fn(self.env_fn, self.num_environments)
        obs_spaces = self.env.get_attr("observation_space")
        act_spaces = self.env.get_attr("action_space")
        if self.obs_space_override is not None:
            self.obs_space_reference = self.obs_space_override
        else:
            self.obs_space_reference = obs_spaces[0]
        self.act_space_reference = act_spaces[0]

        # Create buffer for storing rollout data
        self.buffers = []
        for i in range(self.num_environments):
            self.buffers.append(
                PPOBuffer(
                    self.obs_space_reference.shape, 
                    self.act_space_reference.shape, 
                    self.rollout_length_per_worker, 
                    self.gamma, 
                    self.lam, 
                    self.device
                )
            )

        # Set up tracking of current episode
        try:
            self.env.seed(self.original_seed)
            self.curr_states = self.env.reset()
        except:
            self.curr_states = self.env.reset(seed=self.original_seed)

        # discard the initial info
        self.curr_states = self.curr_states[0]

        self.curr_states = self.state_processing_hook(self.curr_states)
        self.curr_ep_lens = [0]*self.num_environments
        self.curr_ep_rets = [0.0]*self.num_environments
        self.total_steps = 0


    # ==================================================================
    # Data collection

    def rollout(self):
        # collect data for this loop
        for b in self.buffers:
            b.reset_for_collection()

        self.local_epoch_results = []
        t = 0
        rolling = True
        curr_ep_tuples = []
        for i in range(self.num_environments):
            curr_ep_tuples.append([])

        while rolling:
            curr_states_torch = torch.as_tensor(np.array(self.curr_states), dtype=torch.float32)
            if self.device is not None:
                curr_states_torch = curr_states_torch.to(self.device)
            a, v, logp = self.ac.step(curr_states_torch)
            step_output = self.env.step(a)

            next_o, r, d1, d2, info = step_output
            d = [(d1[i] or d2[i]) for i in range(len(d1))]
            next_o = self.state_processing_hook(next_o)

            for i in range(self.num_environments):
                self.curr_ep_lens[i] += 1
                self.curr_ep_rets[i] += deepcopy(float(r[i]))

            # clip rewards (after we have stored the true value)
            if self.clip_rewards:
                r = np.sign(r)

            # isolate per-env data
            for i in range(self.num_environments):
                curr_ep_tuples[i].append(deepcopy((self.curr_states[i], a[i], r[i], v[i], logp[i])))
                self.curr_states[i] = deepcopy(next_o[i])

            # get terminal or timeout status for each env
            terminals = [False]*self.num_environments
            for i in range(self.num_environments):
                terminals[i] = d[i]

            last_step = (t==self.rollout_length_per_worker-1)

            # bulk get any bootstrap values
            values = [0.0]*self.num_environments
            states_to_eval = []
            env_mapping = []
            for i in range(self.num_environments):
                if terminals[i] or last_step:
                    if last_step and not terminals[i]:
                        states_to_eval.append(self.curr_states[i])
                        env_mapping.append(i)

            if len(env_mapping)>0:
                states_torch = torch.as_tensor(np.array(states_to_eval), dtype=torch.float32)
                if self.device is not None:
                    states_torch = states_torch.to(self.device)
                _, vs, _ = self.ac.step(states_torch)
                for i in range(len(env_mapping)):
                    idx = env_mapping[i]
                    values[idx] = vs[i]

            # dump all tuples to rollout callback
            # should this be batched together?
            for i in range(self.num_environments):
                if terminals[i] or last_step:
                    # print(i, "Finished episode with length:", self.curr_ep_lens[i], d1[i], d2[i], last_step)
                    if self.rollout_interception_callback is not None:
                        curr_ep_tuples[i] = self.rollout_interception_callback(curr_ep_tuples[i])

            # save to all buffers
            for i in range(self.num_environments):
                if terminals[i] or last_step:
                    for tup in curr_ep_tuples[i]:
                        self.buffers[i].store(*tup)
                    curr_ep_tuples[i] = []

                    self.buffers[i].finish_path(values[i])

                    if terminals[i]:
                        self.local_epoch_results.append((self.curr_ep_rets[i], self.curr_ep_lens[i], float(t)/self.rollout_length_per_worker))

                        # self.curr_states[i] = self.env.reset() # should be automatic
                        self.curr_ep_lens[i] = 0
                        self.curr_ep_rets[i] = 0.0

            # update loop info: we are tracking "per environment"
            t += 1

            if t>= self.rollout_length_per_worker:
                rolling = False

    # ==================================================================
    # Logging and Saving

    def maybe_gather_results(self):
        return [self.local_epoch_results]

    # ==================================================================
    # Routines to compute and apply updates to the policy network

    def get_and_shuffle_data(self):
        datas = [b.get() for b in self.buffers]
        obs = torch.cat([data['obs'] for data in datas])
        act = torch.cat([data['act'] for data in datas])
        adv = torch.cat([data['adv'] for data in datas])
        logp_old = torch.cat([data['logp'] for data in datas])
        ret = torch.cat([data['ret'] for data in datas])
        perm = np.random.permutation(len(obs))
        self.epoch_data = {
            "obs":obs[perm],
            "act":act[perm],
            "adv":adv[perm],
            "logp":logp_old[perm],
            "ret":ret[perm],
        }
        self.batch = 0