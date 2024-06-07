# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr
# custom modules
from . import minima_search
from kea.broker.probav.adhoc_reader import load


# the first load is to get dask object
da = load()

# [optional] align spatial coordinates with another object such as sentinel-2 data
#"""
def align_spatial(da, ref = '/home/apiwat/git/lstm/lstm_dataset/generator/data/null/2021-10-11-direct/trainSet.nc'):
    s2ds = xr.open_dataset(ref)
    coords = {
        k: np.unique(s2ds.coords[k].values)
        for k in ('latitude', 'longitude')
    }
    coords['latitude'] = np.flip(coords['latitude'])
    return da.interp(
        coords,
        method = 'nearest',
    )
da = align_spatial(da)
#"""

# [optional] resample at the same time grid used to prepare sentinel-2 data for LSTM
#"""
da = da.resample({'time':pd.offsets.Week(weekday=6)}).max()
#"""

# [optional] linear fill NaN along time dimension
"""
da = da.chunk({'time': -1}).interpolate_na('time')
da_ = da.chunk({'time': -1, 'latitude': 100, 'longitude': 100}).interpolate_na('time', use_coordinate = False)
"""

# the second load is to trigger reading from disk and preprocessing
da.load()

# find minimas
minimas = minima_search(
    da, radius = 2,
    min_periods = 1, # this needs to be lowered, down to a minimum of 1, if data has a lot of NaNs.
)

# pick time series at a location to examine the result
pick = {
    k: da.sizes[k] // 2
    for k in ('latitude', 'longitude')
}
print(da.isel(**pick))
print(minimas.isel(**pick))
