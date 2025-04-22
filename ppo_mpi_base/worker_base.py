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

import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import ppo_mpi_base.core as core


# base class to make sure we can easily swap between distributed and centralized modes
class PPOWorkerBase():

    def __init__(
        self, 

        # main setup methods
        env_fn=None,  
        network_fn=None,
        value_network_fn=None,

        # kwargs, to customize envs or networks
        env_kwargs={},
        policy_kwargs={},
        critic_kwargs={},

        # ppo params
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
        device=None, # where to put the networks

        # environment settings
        num_environments=None,
        make_vec_env_fn=None,
        new_gym_interface=True,

        # hooks
        rollout_interception_callback=None, # transform collected episode before storing to buffer
        state_processing_hook=None, # fn(state)->state: transform states as they are observed
        obs_space_override=None, # if the above changes the state size, this is helpful
    ):

        if log_directory[-1] != "/": log_directory += "/"
        if save_location[-1] != "/": save_location += "/"

        if train_batch_size is None:
            train_batch_size = rollout_length_per_worker

        # main setup methods
        self.env_fn = env_fn
        self.network_fn = network_fn
        self.value_network_fn = value_network_fn
        self.env_kwargs = env_kwargs
        self.policy_kwargs = policy_kwargs
        self.critic_kwargs = critic_kwargs

        # ppo settings
        self.original_seed = seed
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
        self.device = device

        # extended env setup
        self.num_environments=num_environments
        self.make_vec_env_fn=make_vec_env_fn
        self.new_gym_interface=new_gym_interface

        # hooks
        self.rollout_interception_callback = rollout_interception_callback
        if state_processing_hook is None:
            self.state_processing_hook = lambda x: x
        else:
            self.state_processing_hook = state_processing_hook
        self.obs_space_override = obs_space_override

        # track performance
        self.best_performance = -100000.0
        self.last_rollout_global_perf = -100000.0
        
        # worker-specific setup
        self.post_init()

        # logger
        if self.rank==0:
            self.summary_writer = SummaryWriter(log_dir=log_directory)

        # build the environment
        self.build_env()

        # build networks, optimizers, etc
        self.build_networks()


    def post_init(self):
        # this must set self.rank and self.world_size
        raise NotImplementedError

    def build_env(self):
        # this must set self.env, self.obs_space_reference, self.act_space_reference
        raise NotImplementedError

    def build_networks(self):
        # Create actor-critic module
        self.ac = core.ActorCritic(
            self.obs_space_reference, 
            self.act_space_reference, 
            self.network_fn, 
            self.value_network_fn, 
            self.policy_kwargs, 
            self.critic_kwargs
        )
        if self.device is not None:
            self.ac.to(self.device)

        # Set up network optimizers
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = torch.optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)



    # ==================================================================
    # Logging and Saving

    def zero_print(self, *args):
	    if self.rank == 0:
	        print(*args)

    def maybe_gather_results(self):
        raise NotImplementedError

    def log(self):
        # collect results across all workers and write to logs
        all_results = self.maybe_gather_results()
        all_results = sum(all_results, [])
        rets = [x[0] for x in all_results]
        lens = [x[1] for x in all_results]
        self.zero_print("Episode Outcomes:", [float(x) for x in rets])

        if self.rank==0 and len(rets)>0:
            dsteps = self.world_size*self.rollout_length_per_worker

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

        self.total_steps += self.world_size*self.rollout_length_per_worker
        if(len(rets)>0):
            self.zero_print("Average Return:", round(np.mean(rets),3), "Total Steps:", f'{self.total_steps:,}')
        else:
            self.zero_print("Average Return: - Total Steps:", f'{self.total_steps:,}')

    def save(self):
        if self.rank==0:

            # only save if new best performance
            best = False
            if self.last_rollout_global_perf >= self.best_performance:
                self.best_performance = self.last_rollout_global_perf
                best = True

            # save (including optimizer states)
            if self.save_location is not None:
                if not os.path.exists(self.save_location):
                    os.makedirs(self.save_location)
                torch.save(self.ac.state_dict(), self.save_location+"model_latest.pt")
                torch.save(self.pi_optimizer.state_dict(), self.save_location+"model_latest_pi_opt.pt")
                torch.save(self.vf_optimizer.state_dict(), self.save_location+"model_latest_vf_opt.pt")

                if best:
                    torch.save(self.ac.state_dict(), self.save_location+"model_best.pt")


    # ==================================================================
    # Routines to compute and apply updates to the policy network

    def zero_policy_grads(self):
        self.pi_optimizer.zero_grad()

    def compute_grads_for_policy_network(self):
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

    def update_policy_network(self):
        if self.pi_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.pi_grad_clip)
        self.pi_optimizer.step()

    # ==================================================================
    # Routines to compute and apply updates to the value network

    def zero_value_grads(self):
        self.vf_optimizer.zero_grad()

    def compute_grads_for_value_network(self):
        data = self.epoch_data
        obs, ret = data['obs'], data['ret']
        obs = obs[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        ret = ret[self.batch*self.train_batch_size:(self.batch+1)*self.train_batch_size]
        v_pred = self.ac.v(obs)
        loss = ((v_pred - ret)**2).mean()
        loss.backward()
        self.batch += 1

    def update_value_network(self):
        if self.v_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.v_grad_clip)
        self.vf_optimizer.step()


    # ==================================================================
    # Hooks for distributed syncing: only used by mpi worker

    def average_grads_for_policy_network(self):
        pass

    def sync_weights_for_policy_network(self):
        pass

    def average_grads_for_value_network(self):
        pass

    def sync_weights_for_value_network(self):
        pass