"""Sequentially load chunks of dask.array.Array

Problem statement:

Imagine a situation where total task composed of

1. loading images from disk
2. reducing data to get final smaller result

Dask scheduler often tries to start up too many loading task
while reducing data is not finished.
This causes memory blow up.

It can be said that dask scheduler does two things wrong here:

1. not ordering task correctly.
2. not having good enough grasp on the memory requirement of each task. / not using estimated result size to help schedule.


(2) maybe solved by specifying abstract resources (a system which dask provided) in order to limit concurrency
# TODO: try resources system.

This module however, attempt to solve issue by forcing chunks of the final result to be
calculated in limited concurrency.
"""
# 3rd party modules
import xarray as xr
import numpy as np
import dask
from dask.distributed import Client, as_completed


def seq_chunk_compute(da):
    """Sequentially compute chunks of dask.array.Array"""
    block_sizes = tuple(len(x) for x in da.chunks)
    results = np.full(block_sizes, None)
    for idx in np.ndindex(block_sizes):
        results[idx] = da.blocks[idx].compute()
    return np.block(results.tolist())


def seq_chunk_load(da):
    """Sequentially load chunks of xarray.DataArray"""
    if not dask.is_dask_collection(da):
        return da
    return da.copy(
        data = seq_chunk_compute(da.data),
        deep = False,
    )


def get_client():
    try:
        return Client.current()
    except ValueError:
        return Client()


def compute_as_complete(tasks, limit = 3):
    """Compute Dask objects in given order with limited concurrency, return in the order they are finished."""
    client = get_client()
    tasks = iter(tasks)
    ac = as_completed([
        client.compute(task)
        for _, task in zip(range(limit), tasks)
    ])
    results = []
    for task in tasks:
        just_completed = next(ac)
        results.append(just_completed.result())
        ac.add(client.compute(task))
    for just_completed in ac:
        results.append(just_completed.result())
    return results


def keyed_compute_as_complete(keys, tasks, limit = 3):
    """Same with `compute_as_complete` but results are returned with keys identifying the task it came from."""
    client = get_client()
    tasks = zip(keys, tasks)
    future2key = {
        client.compute(task): k
        for _, (k, task) in zip(range(limit), tasks)
    }
    ac  = as_completed(future2key.keys())
    results = []
    for k, task in tasks:
        just_completed = next(ac)
        results.append((
            future2key[just_completed],
            just_completed.result(),
        ))
        future = client.compute(task)
        future2key[future] = k
        ac.add(future)
    for just_completed in ac:
        results.append((
            future2key[just_completed],
            just_completed.result(),
        ))
    return results


def limit_chunk_compute(da, limit = 3):
    """Compute dask.array.Array with limited number of chunks processed in parallel."""
    client = get_client()
    block_sizes = tuple(len(x) for x in da.chunks)
    keyed_results = keyed_compute_as_complete(
        np.ndindex(block_sizes),
        (
            da.blocks[idx]
            for idx in np.ndindex(block_sizes)
        ),
        limit,
    )
    results = np.full(block_sizes, None)
    for idx, block in keyed_results:
        results[idx] = block
    return np.block(results.tolist())


def limit_chunk_load(da, limit = 3):
    """Load xarray.DataArray with limited number of chunks processed in parallel."""
    if not dask.is_dask_collection(da):
        return da
    return da.copy(
        data = limit_chunk_compute(da.data, limit = limit),
        deep = False,
    )
