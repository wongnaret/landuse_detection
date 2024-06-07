"""Limit number chunks being written in parallel by `rioxarray.raster_array.RasterArray.to_raster`."""
import re
import math
# 3rd party modules
import dask.distributed
# custom modules
from .seq_chunk_compute import get_client


_store_key_regex = re.compile(
    'store-[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
)


def batch_to_raster(da, *to_raster_args, batch_size = None, **to_raster_kwargs):
    """Submits `rioxarray.raster_array.RasterArray.to_raster` in `batch_size` chunks at a time.
    
    A batch is waited on before continuing to the next batch.
    
    Parameters
    ----------
    da: xarray.DataArray
        Array containing image to write.
    batch_size: int
        Number of chunks being processed in parallel in a batch.
    to_raster_args: list
    to_raster_kwargs: dict
        Positional and keywords arguments for `rioxarray.raster_array.RasterArray.to_raster`.
    """
    client = get_client()
    batch_size = batch_size or nparallel(client = client)
    writejob = da.rio.to_raster(
        *to_raster_args,
        windowed = True,
        lock = dask.distributed.Lock(),
        compute = False,
        **to_raster_kwargs,
    )
    keys = iter(get_store_keys(writejob))
    while True:
        batch = [k for _,k in zip(range(batch_size), keys)]
        if not batch:
            break
        futures = client.get(writejob.dask, batch, sync = False)
        client.gather(futures)
        for f in futures:
            f.release()


def limit_to_raster(da, *to_raster_args, limit = None, **to_raster_kwargs):
    """Submits `rioxarray.raster_array.RasterArray.to_raster` chunks as they completed, limiting chunks processed in parallel.
    
    Parameters
    ----------
    da: xarray.DataArray
        Array containing image to write.
    limit: int
        Number of chunks allowed to be processed in parallel at any one time.
    to_raster_args: list
    to_raster_kwargs: dict
        Positional and keywords arguments for `rioxarray.raster_array.RasterArray.to_raster`.
    """
    client = get_client()
    limit = limit or nparallel(client = client)
    writejob = da.rio.to_raster(
        *to_raster_args,
        windowed = True,
        lock = dask.distributed.Lock(),
        compute = False,
        **to_raster_kwargs,
    )
    keys = iter(get_store_keys(writejob))
    
    # NOTE: similar in structure to `.seq_chunk_compute.compute_as_complete`
    ac = dask.distributed.as_completed(
        client.get(
            writejob.dask,
            [k for _,k in zip(range(limit), keys)],
            sync = False,
        )
    )
    # NOTE: results of writejob tasks are 'normally' None
    for k in keys:
        just_completed = next(ac)
        just_completed.release()
        ac.add(
            client.get(
                writejob.dask,
                k,
                sync = False,
            )
        )
    for just_completed in ac:
        just_completed.release()
    return


def nparallel(factor = 0.5, client = None):
    """Helps calculate how many chunks should be processed in parallel based on number of threads available."""
    client = client or get_client()
    return math.ceil(factor * sum(client.nthreads().values())) + 1


def get_store_keys(writejob):
    """Find the keys to per-chunk storing tasks."""
    layer = get_store_layer(writejob)
    return next(iter(layer.values()))


def get_store_layer(writejob):
    for layer in writejob.dask.layers.values():
        if is_store_layer(layer):
            return layer
    return None


def is_store_layer(layer):
    """Cautiously check if the layer looks like one that perform chunked raster storing job created by `rio.to_raster`."""
    if len(layer) != 1:
        return False
    key, children = next(iter(layer.items()))
    if not (
        isinstance(key, str) and
        _store_key_regex.fullmatch(key) and
        isinstance(children, list) and
        children and
        isinstance(children[0], tuple) and
        isinstance(children[0][0], str) and
        _store_key_regex.fullmatch(children[0][0])
    ):
        return False
    child_key_group = children[0][0]
    return all(child[0] is child_key_group for child in children)
