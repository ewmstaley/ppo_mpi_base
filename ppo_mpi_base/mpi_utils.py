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

import sys
import torch
import numpy as np
from mpi4py import MPI

# mpi utils for pytorch, building from those in from spinning up
# NOTE: when changing pytorch.numpy(), this also changes the tensor


def zero_print(*args):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args)


def average_grads_across_processes(comm, parameters):
    for p in parameters:
        if p.grad is None:
            # print("WARNING: network parameter gradient does not exist.")
            continue

        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(comm, p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def average_grads_across_processes_2(comm, parameters):
    # perform a single mpi average instead of looping through parameters
    all_params = torch.nn.utils.parameters_to_vector([p.grad for p in parameters])
    all_params_numpy = all_params.numpy()
    avg_p_grad = mpi_avg(comm, all_params_numpy)
    all_params_numpy[:] = avg_p_grad[:]


# averages weights across procs, does not retain gradient information
def sync_weights_across_processes(comm, parameters):
    for p in parameters:
        p_numpy = p.data.cpu().numpy()
        comm.Bcast(p_numpy, root=0)



def mpi_avg(comm, x):
    """  average a value across all procs  """
    num_procs = comm.Get_size()
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=MPI.SUM)
    return buff / num_procs


def mpi_std(comm, x):
    """ get the standard deviation across all procs """
    all_values = mpi_sorted_gather(comm, x)
    all_values = np.asarray(all_values)
    return np.std(all_values)


def mpi_avg_filter_nones(comm, x):
    """  average across all procs, but skip None values  """
    if x is not None:
        entry = [np.asarray(x, dtype=np.float32)]
    else:
        entry = []

    collection = comm.allgather(entry)
    filtered_collection = []
    for c in collection:
        if len(c) > 0:
            filtered_collection.append(c[0])

    if len(filtered_collection)==0:
        return None

    avg = np.mean(filtered_collection, axis=0)
    return avg


# NOTE: This sorts data by rank
def mpi_sorted_gather(comm, x):
    """
    collect a list of data across procs, indexed by process rank
    allows None values to be passed which can be filtered later
    these will show up as empty lists, while valid values will be
    length-one lists, i.e. return_value = [[1], [3], [], [1], [], ...]
    """
    rank = comm.Get_rank()
    if x is not None:
        entry = [[rank], [x]]
    else:
        entry = [[rank], []]
    collection = comm.allgather(entry)
    collection.sort(key=lambda v: v[0])
    collection = [v[1] for v in collection]
    return collection


# NOTE: This sorts data itself
def mpi_sorted_fraction(comm, x):
    '''
    collect data across procs, and return the indices of the passed data
    in the globally sorted result
    '''
    rank = comm.Get_rank()
    local_collection = []
    for idx,x in enumerate(x):
        local_collection.append((x, idx, rank))

    # gather all collections
    global_collections = comm.allgather(local_collection)
    global_collection = np.concatenate(global_collections).tolist()

    # sort by first value
    global_collection.sort(key=lambda v: v[0])

    # which entries are ours?
    index_map = []
    for gdx, g in enumerate(global_collection):
        value, idx, r = g
        if r==rank:
            index_map.append((int(idx), gdx))

    index_map.sort(key=lambda v: v[0])

    return index_map, len(global_collection)


def mpi_gather_objects(comm, x, strip_ranks=True):
    """  Collect objects from across processes  """
    rank = comm.Get_rank()
    if x is not None:
        entry = [rank, x]
    else:
        entry = [rank, []]
    collection = comm.allgather(entry)
    collection.sort(key=lambda v: v[0])
    if strip_ranks:
        collection = [v[1] for v in collection]
    return collection


def collect_dict_of_arrays(comm, x):
    """  Collect a dictionary of numpy arrays across processes  """
    collected_dictionaries = mpi_gather_objects(comm, x)
    combined_dictionary = {}
    for k, v in collected_dictionaries[0].items():
        value_array = np.array([]).reshape((0,))
        for dictionary in collected_dictionaries:
            value_array = np.concatenate((value_array, dictionary[k]))
        combined_dictionary[k] = value_array
    return combined_dictionary


# Additional tools employed by spinning up, with comm made explicit here:

def mpi_op(comm, x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(comm, x):
    """  sum a value across all procs  """
    return mpi_op(comm, x, MPI.SUM)


def mpi_statistics_scalar(comm, x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        comm: MPI comm object
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum(comm, [np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(comm, np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(comm, np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(comm, np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def print_now(input_string):
    """  Allows for immediate printing, in particular when running MPI"""
    print(input_string)
    sys.stdout.flush()


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    x = np.asarray([rank, rank+10, rank+100])

    std = mpi_std(comm, x)
    print(std)