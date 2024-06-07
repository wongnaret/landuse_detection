#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: create_dateset.py
 Date Create: 1/11/2021 AD 09:20
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""


from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import netCDF4
import xarray as xr
import click

import datetime


'''
'coastal', 'blue', 'green', 'red', 'veg5', 'veg6', 'veg7', 'nir',
       'narrow_nir', 'water_vapour', 'swir1', 'swir2', 'SCL', 'WVP',
       'AOT'
'''


def sampling_ds(ds, idx):

    coastal = ds['coastal'].sel(location=idx, time=ds.time[:70]) / 10000
    blue = ds['blue'].sel(location=idx, time=ds.time[:70]) / 10000
    green = ds['green'].sel(location=idx, time=ds.time[:70]) / 10000
    red = ds['red'].sel(location=idx, time=ds.time[:70]) / 10000
    veg5 = ds['veg5'].sel(location=idx, time=ds.time[:70]) / 10000
    veg6 = ds['veg6'].sel(location=idx, time=ds.time[:70]) / 10000
    veg7 = ds['veg7'].sel(location=idx, time=ds.time[:70]) / 10000
    nir = ds['nir'].sel(location=idx, time=ds.time[:70]) / 10000
    narrow_nir = ds['narrow_nir'].sel(location=idx, time=ds.time[:70]) / 10000
    water_vapour = ds['water_vapour'].sel(location=idx, time=ds.time[:70]) / 10000
    swir1 = ds['swir1'].sel(location=idx, time=ds.time[:70]) / 10000
    swir2 = ds['swir2'].sel(location=idx, time=ds.time[:70]) / 10000
    SCL = ds['SCL'].sel(location=idx, time=ds.time[:70])
    WVP = ds['WVP'].sel(location=idx, time=ds.time[:70]) / 1000
    AOT = ds['AOT'].sel(location=idx, time=ds.time[:70]) / 1000

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

    return (tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ds.time.values[:70])



@click.command()
#@click.option('--gpu/--cpu', default=False, help=' select cpu or gpu', show_default=True)
#@click.option('--datatype', '-t', type=click.Choice(['RAW', 'RMMEH', 'SHIFT'], case_sensitive=False), default='RMMEH',
#              help='create model for which crop?', show_default=True)
@click.option('--cropclass', '-c', type=click.Choice(['sugarcane', 'rice'], case_sensitive=False), default='sugarcane',
              help='create data for which crop?', show_default=True)
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='input nc file')
@click.option('--output', '-o', default='.', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='ouput directory for result image')
@click.option('--n_samp', '-n', default = 50, show_default=True, help='number of samples for each class')


def start(cropclass, input, output, n_samp):
    seed = 1
    np.random.seed(seed)

    '''
    data = ['../lstm_dataset/20210730_dataset/kpp1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/kpp2_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/central1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/central2_rmmeh_masked/trainSet.nc',
            # '../lstm_dataset/20210730_dataset/central3_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/north_eastern1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/north_eastern2_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/north_eastern3_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/eastern1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20210730_dataset/eastern2_rmmeh_masked/trainSet.nc'
            ]
    '''

    data = ['../lstm_dataset/20211122_dataset/kpp1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/kpp2_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/central1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/central2_rmmeh_masked/trainSet.nc',
            # '../lstm_dataset/20210730_dataset/central3_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/north_eastern1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/north_eastern2_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/north_eastern3_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/eastern1_rmmeh_masked/trainSet.nc',
            '../lstm_dataset/20211122_dataset/eastern2_rmmeh_masked/trainSet.nc'
            ]


    trn_x = None
    trn_y = None
    trn_pos = None
    trn_location = None
    trn_time = None

    test_x = None
    test_y = None
    test_pos = None
    test_location = None
    test_time = None

    for file in data:
        print('read file:', file)
        ds = xr.open_dataset(file)

        band = list(ds.keys())
        band.append('ndvi')
        band.append('ndwi')

        y = ds.y.values

        other_idx_trn = np.random.choice(np.where(y == 0)[0], n_samp)
        sugar_idx_trn = np.random.choice(np.where(y == 1)[0], n_samp)
        rice_idx_trn = np.random.choice(np.where(y == 2)[0], n_samp)

        other_idx_test = np.random.choice(np.where(y == 0)[0], n_samp)
        sugar_idx_test = np.random.choice(np.where(y == 1)[0], n_samp)
        rice_idx_test = np.random.choice(np.where(y == 2)[0], n_samp)

        trn_idx = np.concatenate([other_idx_trn, sugar_idx_trn, rice_idx_trn])
        test_idx = np.concatenate([other_idx_test, sugar_idx_test, rice_idx_test])

        (tmp_array, y, position, location, timestamp) = sampling_ds(ds, trn_idx)
        if trn_x is None:
            trn_x = tmp_array
            trn_y = y
            trn_pos = position
            trn_location = location
            trn_time = timestamp

        # elif len(tmp_array) < max_n_sample:
        else:
            trn_x = np.append(trn_x, tmp_array, axis=0)
            trn_y = np.append(trn_y, y, axis=0)
            trn_pos = np.append(trn_pos, position, axis=0)
            trn_location = np.append(trn_location, location, axis=0)

        (tmp_array, y, position, location, timestamp) = sampling_ds(ds, test_idx)
        if test_x is None:
            test_x = tmp_array
            test_y = y
            test_pos = position
            test_location = location
            test_time = timestamp

        # elif len(tmp_array) < max_n_sample:
        else:
            test_x = np.append(test_x, tmp_array, axis=0)
            test_y = np.append(test_y, y, axis=0)
            test_pos = np.append(test_pos, position, axis=0)
            test_location = np.append(test_location, location, axis=0)

    print("Shape of x:", trn_x.shape)
    print("Shape of y:", trn_y.shape)
    print("Shape of position:", trn_pos.shape)
    print("Shape of location:", trn_location.shape)
    print("Shape of timstamp:", trn_time.shape)

    # Save to NPY format
    np.savez("trainSet3classRMMEH_masked.npz", x=trn_x, y=trn_y, location=trn_location, position=trn_pos,
             timestamp=trn_time, bands=np.array(band))
    np.savez("testSet3classRMMEH_masked.npz", x=test_x, y=test_y, location=test_location, position=test_pos,
             timestamp=test_time, bands=np.array(band))


'''
def start(cropclass, input, output, n_samp):
    ds = xr.open_dataset(input)
    y = ds.y.values

    input_path = Path(input)
    crop_idx = None

    if cropclass == "sugarcane":
        crop_idx = np.random.choice(np.where(y == 1)[0], n_samp)
    elif cropclass == "rice":
        crop_idx = np.random.choice(np.where(y == 2)[0], n_samp)
    
    
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

'''

if __name__ == '__main__':
    start()