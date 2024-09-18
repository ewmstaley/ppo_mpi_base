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

from ppo_mpi_base.worker import PPOWorker
from ppo_mpi_base.timekeeper import TimeKeeper
from ppo_mpi_base.mpi_utils import *
import time

def PPO(
    total_steps,
    env_fn,  
    network_fn,
    value_network_fn,
    seed=0, 
    rollout_length_per_worker=5000, 
    clip_rewards=False,  
    gamma=0.99, 
    lam=0.95,
    max_ep_len=None,
    target_kl=0.01, 
    clip_ratio=0.2,
    entropy_coef=0.0, 
    pi_lr=3e-4,
    vf_lr=1e-3, 
    train_pi_epochs=80, 
    train_v_epochs=80,
    train_batch_size=None,
    pi_grad_clip=1.0,
    v_grad_clip=1.0,
    log_directory="./logs/", 
    save_location="./saved_model/",

    # adding these to support algos like GAIL
    rollout_interception_callback=None,
    external_update_callback=None,
):

    tk = TimeKeeper(total_steps)
    
    worker = PPOWorker(
        env_fn=env_fn,  
        network_fn=network_fn,
        value_network_fn=value_network_fn,
        seed=seed, 
        rollout_length_per_worker=rollout_length_per_worker,  
        gamma=gamma, 
        lam=lam,
        max_ep_len=max_ep_len,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef, 
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        train_batch_size=train_batch_size,
        pi_grad_clip=pi_grad_clip,
        v_grad_clip=v_grad_clip,
        log_directory=log_directory, 
        save_location=save_location,
        rollout_interception_callback=rollout_interception_callback
    )

    # initial weight sync
    worker.sync_weights_for_policy_network()
    worker.sync_weights_for_value_network()

    # how many batches per epoch?
    batches = rollout_length_per_worker//worker.train_batch_size

    # run
    while worker.total_steps < total_steps:

        # collect data
        worker.rollout()
        worker.log()
        tk.log(worker.total_steps)

        # external update, if any
        if external_update_callback is not None:
            external_loss = external_update_callback()
            if external_loss is not None and worker.rank==0:
                worker.summary_writer.add_scalar("external_loss", external_loss, worker.total_steps)

        # update policy network
        for i in range(train_pi_epochs):
            worker.get_and_shuffle_data()
            approx_kls = []
            
            for b in range(batches):
                akl = worker.compute_grads_for_policy_network()
                approx_kls.append(akl)
                worker.average_grads_for_policy_network()
                worker.update_policy_network()

                if mpi_avg(MPI.COMM_WORLD, np.mean(approx_kls)) > 1.5*target_kl:
                    zero_print("Hit KL after", i, "epochs,", b, "batches")
                    break

            if mpi_avg(MPI.COMM_WORLD, np.mean(approx_kls)) > 1.5*target_kl:
                break
            
        worker.sync_weights_for_policy_network()

        # update value network
        for i in range(train_v_epochs):
            worker.get_and_shuffle_data()
            for b in range(batches):
                worker.compute_grads_for_value_network()
                worker.average_grads_for_value_network()
                worker.update_value_network()
        worker.sync_weights_for_value_network()

        # save
        worker.save()

        # report timing estimates
        if worker.rank==0:
            tk.report()