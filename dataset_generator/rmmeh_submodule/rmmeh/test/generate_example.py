"""Generate random example for tests."""
import numpy as np
import xarray as xr


def gen_ex(
    shape = (2, 10, 2),
    dim = 1,
    na_chance = .5,
    low = 0, high = 1,
):
    """Generate random test case.
    
    Parameters
    ----------
    shape: tuple of positive integers
        shape of input array
    dim: int
        dimension to perform smoothing along
    na_chance: float in [0,1]
        chance of nan at each location
    low,high: float or int
        range of uniform random value
    
    Returns
    -------
    test_case: dictionary
        dictionary of all parameters used together with
        key 'a' mapped to the generated random array.
    """
    a = np.random.uniform(low = low, high = high, size = shape)
    a[np.random.rand(*shape) <= na_chance] = np.nan
    return locals()


def dsk_gen_ex(
    shape = (10000,30,10000),
    dim = 1,
    na_chance = .5,
    chunks = 'auto',
):
    from dask.array.random import random as dar
    from dask.array import where
    a = where(
        dar(shape, chunks = chunks) > na_chance,
        dar(shape, chunks = chunks),
        np.nan,
    )
    return locals()


def xr_gen_ex(*args, dim = 1, chunks = 'auto', **kwargs):
    if chunks:
        dic = dsk_gen_ex(*args, dim = dim, chunks = chunks, **kwargs)
    else:
        dic = gen_ex(*args, dim = dim, **kwargs)
    dic['a'] = xr.DataArray(
        dic['a']
    ).rename({f'dim_{dim}': 'time'})
    dic['dim'] = 'time'
    return dic
