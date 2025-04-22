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

from ppo_mpi_base.mpi_worker import PPOMPIWorker
from ppo_mpi_base.vec_worker import PPOVecWorker
from ppo_mpi_base.timekeeper import TimeKeeper
from ppo_mpi_base.mpi_utils import *
import time

def PPO(
    total_steps,

    rollout_length_per_worker=512,
    train_pi_epochs=40, 
    train_v_epochs=40, 
    target_kl=0.0,
    use_vectorized_envs=False,

    grad_accum=1,
    external_update_callback=None, # hook between rollout and update
    delay_training_for_steps=0,

    **worker_kwargs # almost all args are passed through to worker. See worker_base.py.
):

    tk = TimeKeeper(total_steps)
    
    worker_cls = PPOVecWorker if use_vectorized_envs else PPOMPIWorker
    worker_kwargs["rollout_length_per_worker"] = rollout_length_per_worker
    worker = worker_cls(**worker_kwargs)

    # initial weight sync
    worker.sync_weights_for_policy_network()
    worker.sync_weights_for_value_network()

    # how many batches per epoch?
    batches = rollout_length_per_worker//worker.train_batch_size
    if use_vectorized_envs:
        batches = batches*worker.num_environments

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

        if worker.total_steps < delay_training_for_steps:
            continue

        # update policy network
        for i in range(train_pi_epochs):
            worker.get_and_shuffle_data()
            approx_kls = []
            
            worker.zero_policy_grads()
            for b in range(batches):
                akl = worker.compute_grads_for_policy_network()
                approx_kls.append(akl)

                if b%grad_accum == 0:
                    worker.average_grads_for_policy_network()
                    worker.update_policy_network()
                    worker.zero_policy_grads()

                if mpi_avg(MPI.COMM_WORLD, np.mean(approx_kls)) > 1.5*target_kl:
                    zero_print("Hit KL after", i, "epochs,", b, "batches")
                    break

            if mpi_avg(MPI.COMM_WORLD, np.mean(approx_kls)) > 1.5*target_kl:
                break
            
        worker.sync_weights_for_policy_network()

        # update value network
        for i in range(train_v_epochs):
            worker.get_and_shuffle_data()
            worker.zero_value_grads()
            for b in range(batches):
                worker.compute_grads_for_value_network()

                if b%grad_accum == 0:
                    worker.average_grads_for_value_network()
                    worker.update_value_network()
                    worker.zero_value_grads()

        worker.sync_weights_for_value_network()

        # save
        worker.save()

        # report timing estimates
        if worker.rank==0:
            tk.report()