import xarray as xr
import numpy as np
import pandas as pd


def dict2xr(ds):
    ds_ = xr.Dataset(
        data_vars = {
            'x': xr.DataArray(
                data = ds['x'],
                dims = ds['xdims'],
                coords = {
                    k: v
                    for k,v in ds.items()
                    if k in ds['xdims']
                },
            ),
            'y': xr.DataArray(
                data = ds['y'],
                dims = ['location'],
            ),
        },
        coords = {
            'weeks': ('time',ds['weeks']),
            'row': ('location',ds['row']),
            'col': ('location',ds['col']),
        }
    )
    # try to infer frequency of time index
    if 'time' in ds_.indexes:
        ds_.indexes['time'].freq = ds_.indexes['time'].inferred_freq
    # restore location as multiindex
    ds_['location'] = pd.MultiIndex.from_tuples(ds_.location.values, names=['latitude','longitude'])
    return ds_

def read(filename):
    return dict2xr(
        dict(np.load(filename, allow_pickle = True))
    )

if __name__ == '__main__':
    import sys
    path = (
        sys.argv[1]
        if len(sys.argv) > 1 else
        '../generator/kc_full_data/trainSet.npz'
    )
    ds = read(path)
