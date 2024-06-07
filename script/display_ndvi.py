#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: display_ndvi.py
 Date Create: 16/9/2021 AD 12:18
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""


import click
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

import datetime

seed = 1
np.random.seed(seed)


def sampling_ds(ds, idx):
    coastal = ds['coastal'].sel(location=idx) / 10000
    blue = ds['blue'].sel(location=idx) / 10000
    green = ds['green'].sel(location=idx) / 10000
    red = ds['red'].sel(location=idx) / 10000
    veg5 = ds['veg5'].sel(location=idx) / 10000
    veg6 = ds['veg6'].sel(location=idx) / 10000
    veg7 = ds['veg7'].sel(location=idx) / 10000
    nir = ds['nir'].sel(location=idx) / 10000
    narrow_nir = ds['narrow_nir'].sel(location=idx) / 10000
    water_vapour = ds['water_vapour'].sel(location=idx) / 10000
    swir1 = ds['swir1'].sel(location=idx) / 10000
    swir2 = ds['swir2'].sel(location=idx) / 10000
    SCL = ds['SCL'].sel(location=idx)
    WVP = ds['WVP'].sel(location=idx) / 1000
    AOT = ds['AOT'].sel(location=idx) / 1000

    ndvi = (nir - red) / (nir + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)

    y = ds.y.sel(location=idx)

    tmp_array = np.array([coastal, blue, green, red, veg5, veg6, veg7, nir,
                          narrow_nir, water_vapour, swir1, swir2, SCL, WVP, AOT,
                          ndvi, ndwi])

    i = ds.row.sel(location=idx)
    j = ds.col.sel(location=idx)
    position = np.array([i, j])

    lat = ds.latitude.sel(location=idx)
    lon = ds.longitude.sel(location=idx)
    location = np.array([lat, lon])

    return (tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ds.time.values)


def sampling_shifted(ds, idx):
    coastal = ds.x.sel(band='coastal', location=idx) / 10000
    blue = ds.x.sel(band='blue', location=idx) / 10000
    green = ds.x.sel(band='green', location=idx) / 10000
    red = ds.x.sel(band='red', location=idx) / 10000
    veg5 = ds.x.sel(band='veg5', location=idx) / 10000
    veg6 = ds.x.sel(band='veg6', location=idx) / 10000
    veg7 = ds.x.sel(band='veg7', location=idx) / 10000
    nir = ds.x.sel(band='nir', location=idx) / 10000
    narrow_nir = ds.x.sel(band='narrow_nir', location=idx) / 10000
    water_vapour = ds.x.sel(band='water_vapour', location=idx) / 10000
    swir1 = ds.x.sel(band='swir1', location=idx) / 10000
    swir2 = ds.x.sel(band='swir2', location=idx) / 10000
    SCL = ds.x.sel(band='SCL', location=idx)
    WVP = ds.x.sel(band='WVP', location=idx) / 1000
    AOT = ds.x.sel(band='AOT', location=idx) / 1000

    ndvi = (nir - red) / (nir + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)

    y = ds.y.sel(location=idx)

    tmp_array = np.array([coastal, blue, green, red, veg5, veg6, veg7, nir,
                          narrow_nir, water_vapour, swir1, swir2, SCL, WVP, AOT,
                          ndvi, ndwi])

    i = ds.row.sel(location=idx)
    j = ds.col.sel(location=idx)
    position = np.array([i, j])

    lat = ds.latitude.sel(location=idx)
    lon = ds.longitude.sel(location=idx)
    location = np.array([lat, lon])

    return (tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ds.crop_age.values)

@click.command()
#@click.option('--gpu/--cpu', default=False, help=' select cpu or gpu', show_default=True)
@click.option('--datatype', '-t', type=click.Choice(['RAW', 'RMMEH', 'SHIFT'], case_sensitive=False), default='RMMEH',
              help='create model for which crop?', show_default=True)
@click.option('--cropclass', '-c', type=click.Choice(['other', 'sugarcane', 'rice', 'all'], case_sensitive=False), default='RMMEH',
              help='create model for which crop?', show_default=True)
@click.option('--input', '-f', default='', type=click.Path(exists=True), help='input nc file')
@click.option('--output', '-o', default='.', type=click.Path(exists=True, file_okay=False, dir_okay=True,), help='ouput directory for result image')
@click.option('--n_samp', '-n', default = 50, show_default=True, help='number of samples for each class')

def start(datatype, cropclass, input, output, n_samp):
    ds = xr.open_dataset(input)
    y = ds.y.values

    input_path = Path(input)
    crop_idx = None

    if cropclass == "other":
        crop_idx = np.random.choice(np.where(y == 0)[0], n_samp)
    elif cropclass == "sugarcane":
        crop_idx = np.random.choice(np.where(y == 1)[0], n_samp)
    elif cropclass == "rice":
        crop_idx = np.random.choice(np.where(y == 2)[0], n_samp)
    else :
        crop_idx = np.random.choice(range(len(y)), n_samp)

    if datatype == "RAW":
        (ndvi, y, position, location, timestamp) = sampling_ds(ds, crop_idx)
    elif datatype == "RMMEH":
        (ndvi, y, position, location, timestamp) = sampling_ds(ds, crop_idx)
    elif datatype == "SHIFT":
        (ndvi, y, position, location, timestamp) = sampling_shifted(ds, crop_idx)

    #PLOT
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    fig.set_size_inches(18, 3)
    ax = plt.axes()
    ax.set_ylim(top=1)
    for idx in range(n_samp):
        ax.plot(timestamp, ndvi[idx, -2, :], color='grey');

    plt.xticks(rotation=70, size=7)
    #plt.show()
    plt.savefig(output+'/%s_%s_%s_%d.png'%(datatype, cropclass, input_path.parts[-2], n_samp), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    start()