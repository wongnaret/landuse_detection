# 3rd party modules
import xarray as xr
from dask.distributed import Client
import dask.array
# custom modules
from . import seq_chunk_load, limit_chunk_load
from ...test.generate_example import xr_gen_ex
from ...xr_rmmeh import rmmeh
from ...nd_rmmeh.apply2xarray import apply_rmmeh


def test_create_random_array():
    n = 10000
    da = xr.DataArray(
        data = dask.array.random.random((n,n), chunks = 'auto'),
    )
    breakpoint()
    ret = seq_chunk_load(da)
    # We expect to see chunks are not processed in parallel
    return locals()


def test_limit_create_random_array(limit = 3):
    n = 10000
    da = xr.DataArray(
        data = dask.array.random.random((n,n), chunks = 'auto'),
    )
    breakpoint()
    ret = limit_chunk_load(da, limit = limit)
    # We expect to see at most `limit` chunks processed in parallel.
    return locals()

    
def test_xr_rmmeh():
    a = xr_gen_ex(shape = (5000,30,5000), dim = 1, chunks = 'auto')['a']
    result = rmmeh(a, 'time', hanning_window = 11)
    breakpoint()
    loaded = seq_chunk_load(result)
    # We expect to see chunks of result are scheduled sequentially.
    return locals()


def test_nd_rmmeh():
    a = xr_gen_ex(shape = (5000,30,5000), dim = 1, chunks = 'auto')['a']
    result = apply_rmmeh(a, dim = 'time', hanning_window = 11)
    breakpoint()
    loaded = seq_chunk_load(result)
    # We expect to see chunks of result are scheduled sequentially.
    return locals()


# TODO: assert result from normal load equals limit_chunk_load
def test_limit_nd_rmmeh(limit = 3):
    a = xr_gen_ex(shape = (5000,30,5000), dim = 1, chunks = 'auto')['a']
    result = apply_rmmeh(a, dim = 'time', hanning_window = 11)
    breakpoint()
    loaded = limit_chunk_load(result)
    # We expect to see at most `limit` chunks processed in parallel.
    return locals()


if __name__ == '__main__':
    client = Client()
    print('Available tests:')
    for k in list(globals().keys()):
        if k.startswith('test_'):
            print(k)
