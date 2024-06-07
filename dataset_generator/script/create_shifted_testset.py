#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: create_shifted_testset.py
 Date Create: 24/11/2021 AD 09:11
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
import netCDF4
import xarray as xr

import datetime

from skimage.io import imshow, imread
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import histogram, cumulative_distribution, equalize_hist
from skimage import img_as_ubyte, img_as_uint
from pathlib import Path

from shift.v2.shift import *

import cv2

import click

seed = 1
np.random.seed(seed)

@click.command()
#@click.option('--gpu/--cpu', default=False, help=' select cpu or gpu', show_default=True)
#@click.option('--datatype', '-t', type=click.Choice(['RAW', 'RMMEH', 'SHIFT'], case_sensitive=False), default='RMMEH',
#              help='create model for which crop?', show_default=True)
@click.option('--cropclass', '-c', type=click.Choice(['sugarcane', 'rice', 'all'], case_sensitive=False), default='sugarcane',
              help='create data for which crop?', show_default=True)
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='input nc file')
@click.option('--output', '-o', default='.', type=click.Path(file_okay=True, dir_okay=False), help='output file for result image')
@click.option('--n_samp', '-n', default = 50, show_default=True, help='number of samples for each class')


def start(cropclass, input, output, n_samp):



    age = {
        'sugarcane':53,
        'rice': 17,
        'all': 53
    }

    crop_label = {
        'sugarcane': 1,
        'rice': 2,
        'all': 0
    }

    others = {
        'sugarcane': [2, 3],
        'rice': [3],
    }

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


    print('read file:', input)
    ds = xr.open_dataset(input)

    dss = DatasetSlicer(input)

    print("%s Sampling %d pixels per class......"%(output,n_samp))
    target_batch_trn = dss.get_batch(
        label=crop_label[cropclass],
        age=age[cropclass],
        ntimestep=106,
        nlocation=n_samp,
    )

    target_batch_test = dss.get_batch(
        label=crop_label[cropclass],
        age=age[cropclass],
        ntimestep=106,
        nlocation=n_samp,
    )

    random_batch_trn = dss.get_random_slice_batch(
        label=others[cropclass],
        ntimestep=106,
        nlocation=n_samp,
    )
    random_batch_test = dss.get_random_slice_batch(
        label=others[cropclass],
        ntimestep=106,
        nlocation=n_samp,
    )

    #train_target
    print("Load training-target batch for %s"%output)
    (tmp_array, y, position, location, timestamp) = prepare_ds(target_batch_trn, False)
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


    #train_other
    print("Load training-random batch for %s"%output)
    (tmp_array, y, position, location, timestamp) = prepare_ds(random_batch_trn, True)
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


    #test_target
    print("Load testing-target batch for %s"%output)
    (tmp_array, y, position, location, timestamp) = prepare_ds(target_batch_test, False)
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


    #test_other
    print("Load testing-random batch for %s"%output)
    (tmp_array, y, position, location, timestamp) = prepare_ds(random_batch_test, True)
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

    band = list(ds.keys())
    band.append('ndvi')
    band.append('ndwi')
    band.append('ndbi')

    print("Shape of x:", trn_x.shape)
    print("Shape of y:", trn_y.shape)
    print("Shape of position:", trn_pos.shape)
    print("Shape of location:", trn_location.shape)
    print("Shape of timstamp:", trn_time.shape)


    dir_path = Path(output).parent
    filename = Path(output).name

    # Save to NPY format
    np.savez("%s/trainSet_%s_%s.npz"%(dir_path, filename, cropclass), x=trn_x, y=trn_y, location=trn_location, position=trn_pos, timestamp=trn_time,
             bands=np.array(band))
    np.savez("%s/testSet_%s_%s.npz"%(dir_path, filename, cropclass), x=test_x, y=test_y, location=test_location, position=test_pos,
             timestamp=test_time, bands=np.array(band))

    print("%s Data is saved."%output)

def prepare_ds(ds, is_random):

    coastal = ds['coastal'].compute()
    coastal /= 10000

    blue = ds['blue'].compute()
    blue /= 10000

    green = ds['green']
    green /= 10000

    red = ds['red'].compute()
    red /= 10000

    veg5 = ds['veg5'].compute()
    veg5 /= 10000

    veg6 = ds['veg6'].compute()
    veg6 /= 10000

    veg7 = ds['veg7'].compute()
    veg7 /= 10000

    nir = ds['nir'].compute()
    nir /= 10000

    narrow_nir = ds['narrow_nir'].compute()
    narrow_nir /= 10000

    water_vapour = ds['water_vapour'].compute()
    water_vapour /= 10000

    swir1 = ds['swir1'].compute()
    swir1 /= 10000

    swir2 = ds['swir2'].compute()
    swir2 /= 10000

    SCL = ds['SCL'].compute()

    WVP = ds['WVP'].compute()
    WVP /= 1000

    AOT = ds['AOT'].compute()
    AOT /= 1000

    ndvi = (nir - red) / (nir + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)
    ndbi = (swir1 - red) / (swir1 + red + .00001)

    #y = ds.y


    tmp_array = np.array([coastal, blue, green, red, veg5, veg6, veg7, nir,
                          narrow_nir, water_vapour, swir1, swir2, SCL, WVP, AOT,
                          ndvi, ndwi, ndbi])

    i = ds.row
    j = ds.col
    position = np.array([i, j])


    if is_random:
        y = np.zeros_like(ds.sample)
        lat = ds.latitude
        lon = ds.longitude
        location = np.array([lat, lon])
        return (tmp_array.transpose((1, 0, 2)), y, position.transpose(), location.transpose(), ds.time.values*7)
    else:
        y = np.ones_like(ds.sample)
        #location = ds.location.values
        tmp = np.asarray([sublist for sublist in ds.location.values])
        location = np.array([tmp[:, 0], tmp[:, 1]])
        return (tmp_array.transpose((1, 0, 2)), y, position.transpose(), location.transpose(), ds.age.dt.days.values)


if __name__ == '__main__':
    start()