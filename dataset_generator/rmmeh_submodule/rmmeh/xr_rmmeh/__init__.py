"""Reimplementation of gRMMEH that uses xarray directly as much as possible.

Hopefully, this will help dask better manage
intermediate values in its scheduling.
"""
import numpy as np
import bottleneck
import xarray as xr


_default_hanning_window = 5


def avg_neighbor(obj, dim = 'time'):
    """Find average of neighboring values along given dimension.
    
    obj: xarray.DataArray or xarray.Dataset
    dim: str
        dimension name.
    """
    return (obj.shift(time = 1) + obj.shift(time = -1))/2


# TODO: try bottleneck for speed up (or check if xarray already is using it)
def rmmeh(
    obj, dim,
    median_window = 5, hanning_window = _default_hanning_window,
    use_max = True, # linear_fill_max_gap = 0,
):
    """N-dimensional version of RMMEH, optimised for performance on xarray with dask scheduler.
    
    obj: xarray.DataArray or xarray.Dataset
        must not contain dimension named '_choice'.
    dim: str
        dimension name
    median_window -- positive odd integer
        The size of window used for calculation of running median.
    hanning_window -- positive odd integer
        The size of hanning window used to do the final weighted moving average.
    use_max -- Boolean
        Use maximum to conbine (original, running medians,average neighbor).
        Otherwise, use minimum.
    """
    an = avg_neighbor(obj, dim)
    medians = obj.rolling(
        {dim: median_window},
        min_periods = 1,
        center = True,
    ).median()
    maxed = xr.concat([obj, an, medians], '_choice')
    maxed = getattr(maxed, 'max' if use_max else 'min')('_choice', skipna = True)
    return nanhann(maxed, dim, hanning_window)

    
def nanhann(
    obj, dim,
    hanning_window = _default_hanning_window,
):
    """Hann filter along a dimension, ignoring NaNs.
    
    obj: xarray.DataArray or xarray.Dataset
        must not contain dimension named '_window'.
    dim: str
        dimension name
    hanning_window -- positive odd integer
        The size of hanning window used to do the final weighted moving average.
    """
    if hanning_window <= 3:
        return obj
    window = xr.DataArray(np.hanning(hanning_window)[1:-1], dims=['_window'])
    isnan = np.isnan(obj)
    total = xr.where(isnan, 0, obj).rolling(
        time = window.sizes['_window'],
        center = True,
    ).construct(
        time = '_window',
        fill_value = 0,
    ).dot(window)
    denom = (~isnan).rolling(
        time = window.sizes['_window'],
        center = True,
    ).construct(
        time = '_window',
        fill_value = 0,
    ).dot(window)
    return total / denom
