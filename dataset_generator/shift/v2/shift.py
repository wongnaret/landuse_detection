"""Generates datasets aligned by plant age.

New shifting requirement created on 2021-11-05:

1. รับ Random seed
2. ระบุว่า class อะไร (ข้าว, อ้อย)
3. ระบุอายุ n เดือน
4. ระบุจำนวน sampling (n sample, -1 = ทั้งหมด)
5. Positive class ใช้ proba-v เป็น guide ในการหาวันปลูก แล้ว extract ข้อมูลอายุพืชตามนั้น
6. Negative class ไม่สนวันปลูก และ สุ่มตัดข้อมูล n เดือน ที่ตำแหน่งไหนก็ได้ออกมาจาก profile
7. กรณีของข้าว ในหนึ่งปีจะปลูกได้หลายรอบเอาหลายรอบการผลิตมารวมกันได้
"""
import sys
from pathlib import Path
import logging
import datetime
# 3rd party modules
import click
import numpy as np
import pandas as pd
import xarray as xr
# custom modules
sys.path.insert(0,str(Path(__file__).parent))
from open_dataset_default_auto_chunk import open_dataset_default_auto_chunk
sys.path.insert(0,str(Path(__file__).parent.joinpath('../..')))
from minima_search import minima_search


# https://github.com/pallets/click/issues/405#issuecomment-470812067
class ClickPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


# TODO: use numpy's new `default_rng`
# see: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
class DatasetSlicer:
    
    def __init__(
        self, inputs, seed = None,
        probav_minima_radius = 6
    ):
        """Yields sliced version of input dataset.
        
        inputs: list of str or Path
            Paths to .nc files
        seed: int or None
            Random seed.
        probav_minima_radius: int
            Search radius (timesteps) for minima in probav signal (which is 10 days per timestep).
        """
        self.set_seed(seed)
        self.inputs = inputs
        self.ds = open_dataset_default_auto_chunk(inputs, chunks = {'time': -1})
        self.ds = self.ds.set_index({'location': ['latitude','longitude']})
        self.ndvi = do_ndvi(self.ds)
        #
        pvds = load_probav_like(self.ds)
        pvds = sharpen(pvds, 'time')
        pvm = minima_search(pvds, probav_minima_radius, tiebreak = 'high')
        pvdf = pd.DataFrame(
            np.argwhere(pvm.values),
            columns = pvm.dims,
        )
        for x in ('time', 'latitude','longitude'):
            pvdf[x] = pvm.coords[x].values[pvdf[x]]
        pvdf = pvdf.groupby(['latitude', 'longitude']).apply(lambda x: pd.Index(x.time)).sort_index().unstack()
        self.pvdf = pvdf
    
    def set_seed(self, seed):
        # record state before setting by seed, remeber what state the seed gives, restore state
        pre = np.random.get_state()
        np.random.seed(seed)
        self.rstate = np.random.get_state()
        np.random.set_state(pre)
    
    def use_seed(self):
        # used stored seed if exists, then empty the stored seed
        if self.rstate is not None:
            np.random.set_state(self.rstate)
        self.rstate = None
    
    def get_iter(
        self,
        label, ntimestep, nlocation = None, age = None,
        radius = 8, ncycle = 2,
        method = 'nearest', tolerance = None,
    ):
        """Get a batch of sliced dataset.
        
        label: int or list of int
            Integer label of class to produce data of.
        ntimestep: int
            Size of output time dimension.
        nlocation: int or None
            Size of output location dimension.
            None or -1 means all locations.
        age: int or None
            The age of plant at the end position of returned profiles.
            The age is measured in number of timesteps.
            In other words, it is the number of timesteps from the ending of profile
            backwards to (but excluding) crop start position.
            If None, defaults to `ntimestep - 1`.
        radius: int
            radius to search for minima NDVI
        method: str
            Method to locate S2 minimas using PROBAV minimas as indexer.
            See `pandas.Index.get_indexer`.
            Example:
            - If 'nearest', PROBAV LAI minimas will be used to locate the closest S2 NDVI minimas.
            - If 'ffill', PROBAV LAI minimas will be used to search backwards for the closest S2 NDVI minimas.
        tolerance: timedelta
            The maximum time difference between minimas of PROBAV and S2 that can be considered matching.
        """
        if age is None:
            age = ntimestep - 1
        if isinstance(label, int):
            label = [label]
        self.use_seed()
        ndvi = self.ndvi.sel(location = self.ndvi.y.isin(label))
        if nlocation < 0:
            nlocation = None
        elif nlocation is not None:
            ndvi = ndvi.isel(location = np.sort(
                np.random.choice(ndvi.sizes['location'], nlocation, replace = False)
            ))
        logging.info('Loading from disk and searching for S2 NDVI minimas.')
        minima = minima_search(ndvi, radius).load()
        # reshuffle order after loading in order
        if nlocation is not None:
            minima = minima.isel(location = np.random.permutation(minima.sizes['location']))
        for (lat,lon),ts in minima.groupby('location'):
            # compress the S2 minima series
            ms = ts.time[ts.squeeze()].indexes['time']
            # find the corresponding probav minima series
            pvms = self.pvdf.iloc[
                self.pvdf.index.get_loc(lat, method = 'nearest'),
                self.pvdf.columns.get_loc(lon, method = 'nearest'),
            ]
            # locate S2 minimas by PROBAV minimas
            idx = ms.get_indexer(
                pvms,
                method = method,
                tolerance = tolerance, # radius * ?,
            )
            # discard -1 index which indicates no match, discard duplicates, and trim to ncycle
            idx = idx[idx != -1]
            ms = ms[idx].unique()[:ncycle]
            # for each starting point found, yield slice of self.ds
            for crop_start in ms:
                # the index to stop taking profile of
                istop = self.ds.indexes['time'].get_loc(crop_start) + age + 1
                # skip this crop_start if missing data any place in the profile
                if (istop >= self.ds.sizes['time']) or (ntimestep > istop):
                    continue
                # select time series for this location
                locts = self.ds.sel(location = (lat,lon)).isel(time = slice(istop - ntimestep, istop))
                locts = locts.assign_coords(crop_start = crop_start)
                locts = locts.drop_vars('time').rename_dims(time = 'age').assign_coords(
                    age = ('age', locts.indexes['time'] - crop_start)
                )
                #logging.info('Yielding timeseries for {} crop_start {} age {}'.format((lat,lon), crop_start, age))
                yield locts
    
    def get_batch(self, *args, **kwargs):
        logging.info('Producing age detected batch.')
        return xr.concat(
            self.get_iter(*args, **kwargs),
            'sample',
        )
    
    def get_random_slice_iter(
        self, label, ntimestep, nlocation = None,
    ):
        """Get a batch of randomly sliced dataset.
        
        label: int or list of int
            Integer label of class to produce data of.
        ntimestep: int
            Size of output time dimension.
        nlocation: int or None
            Size of output location dimension.
            None or -1 means all locations.
        """
        # TODO: this could be made faster with (pure) numpy advance indexing.
        if self.ds.sizes['time'] < ntimestep:
            raise ValueError('requested larger ntimestep than available')
        if isinstance(label, int):
            label = [label]
        # filter by label first
        ds = self.ds.sel(location = self.ds.y.isin(label))
        # random location sample
        if nlocation is not None and nlocation >= 0:
            ds = ds.isel(
                location = np.random.choice(ds.sizes['location'], nlocation)
            )
        for (lat,lon),ts in ds.groupby('location'):
            start = np.random.randint(ts.sizes['time'] - ntimestep + 1)
            yield ts.drop_vars('time')\
                .reset_index('location').rename_dims(location = 'sample')\
                .isel(time = slice(start, start + ntimestep))
    
    def get_random_slice_batch(self, *args, **kwargs):
        logging.info('Producing random slice batch.')
        return xr.concat(
            self.get_random_slice_iter(*args, **kwargs),
            'sample',
        )


def load_probav_like(
    ref,
    time_buffer = datetime.timedelta(days = 60),
    latitude_buffer = 0.003,
    longitude_buffer = 0.003,
    linear_fill = True,
):
    """Load probav into memory bounded by coordinates like a given reference object,
    
    ref: xarray.DataArray or xarray.Dataset
        must contain dimension 'time' and coordinates 'latitude' and 'longitude'.
    x_buffer:
        size along 'x' dimension to overshoot when loading.
    """
    logging.info('Loading probav LAI as minima search guide.')
    from kea.broker.probav.adhoc_reader import load as probav_load
    ds = probav_load(date_slice = slice(
        ref.indexes['time'].min().date() - time_buffer,
        ref.indexes['time'].max().date() + time_buffer,
    ))
    ds = ds.sel(
        latitude = slice(
            ref.latitude.max().item() + latitude_buffer,
            ref.latitude.min().item() - latitude_buffer,
        ),
        longitude = slice(
            ref.longitude.min().item() - longitude_buffer,
            ref.longitude.max().item() + longitude_buffer,
        ),
    )
    ds.load()
    logging.info('Finished loading probav LAI.')
    if linear_fill:
        ds = ds.interpolate_na('time', use_coordinate = False)
    logging.info('Finished linear filling probav LAI.')
    return ds


def do_ndvi(ds):
    logging.info('Retriving NDVI')
    return ds.data_vars.get(
        'ndvi',
        (ds.nir - ds.red) / (ds.nir + ds.red)
    )


def sharpen(da, dim = 'time', kernel1d = [-1,3,-1]):
    """Increase contrast along one dimension.
    
    da: xr.DataArray
    dim: str
        Dimension name to increase constrast along.
    """
    import scipy.signal
    kernel = np.expand_dims(
        kernel1d,
        axis = [i for i,x in enumerate(da.dims) if x != dim]
    )
    r = scipy.signal.convolve(
        da.values,
        kernel,
        method = 'direct',
        mode = 'same',
    )
    return da.copy(data = r)
