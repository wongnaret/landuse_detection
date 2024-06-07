"""Reads netCDF back into xarray.Dataset representing

Most of the works are already done by xarray.open_dataset,
we only restore the few properties xarray currently does not serialise well into netCDF file.
Such properties are:
- multi-index
- frequency
"""
import xarray as xr
import pandas as pd


def read(filename, xy_vars = False):
    """
    xy_vars: bool
        whether to concat all bands together as a single 'x' variable
        and transform 'y' from being a coordinate to a variable.
    """
    ds = xr.open_dataset(filename)
    ds = adjust(ds)
    if xy_vars:
        ds = convert_to_xy_vars(ds)
    return ds


def adjust(ds):
    ds = ds.copy()
    # try to infer frequency of time index
    if 'time' in ds.indexes:
        ds.indexes['time'].freq = ds.indexes['time'].inferred_freq
    # restore location as multi-index
    ds = ds.set_index({'location': ['latitude','longitude']})
    return ds


def convert_to_xy_vars(ds):
    # TODO: somehow xr.concat force load data, it shouldn't need to
    return xr.concat(
        ds.data_vars.values(),
        'band',
    ).rename('x').transpose('location','band','time').to_dataset().reset_coords(['y']).assign_coords(
        band = ('band',list(ds.data_vars.keys()))
    )