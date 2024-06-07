#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: probav_shifted.py
 Date Create: 2/11/2021 AD 14:04
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

import click
import numpy as np
import xarray as xr

# from torch import nn
# from torch.nn import functional as F

# from LSTMClassifier import LSTMClassifier
# from lstm_utils import loops_fill, numpy_fill, convert_to_df, create_datasets, create_loaders, CyclicLR, cosine

data = ['kpp1_rmmeh_masked/trainSet.nc',
        'kpp2_rmmeh_masked/trainSet.nc',
        'central1_rmmeh_masked/trainSet.nc',
        'central2_rmmeh_masked/trainSet.nc',
        # '../lstm_dataset/20210730_dataset/central3_rmmeh_masked/trainSet.nc',
        'north_eastern1_rmmeh_masked/trainSet.nc',
        'north_eastern2_rmmeh_masked/trainSet.nc',
        'north_eastern3_rmmeh_masked/trainSet.nc',
        'eastern1_rmmeh_masked/trainSet.nc',
        'eastern2_rmmeh_masked/trainSet.nc'
        ]

date_slice_sugar = [slice('2018-11-01', '2019-07-30'),
                    slice('2018-11-01', '2019-07-30'),
                    slice('2019-01-01', '2019-06-30'),
                    slice('2019-01-01', '2019-06-30'),
                    slice('2019-01-01', '2019-06-30'),
                    slice('2019-01-01', '2019-08-30'),
                    slice('2019-01-01', '2019-06-30'),
                    slice('2018-11-01', '2019-05-30'),
                    slice('2019-01-01', '2019-05-30'),
                    ]

date_slice_rice = [slice('2018-11-01', '2019-03-30'),
                   slice('2018-11-01', '2019-03-30'),
                   slice('2019-06-01', '2019-07-30'),
                   slice('2019-06-01', '2019-07-30'),
                   slice('2019-03-01', '2019-07-30'),
                   slice('2019-05-01', '2019-07-30'),
                   slice('2019-03-01', '2019-07-30'),
                   slice('2019-01-01', '2019-07-30'),
                   slice('2019-01-01', '2019-07-30'),
                   ]


def sampling_ds(ds, idx, planting_slice):

    nir = ds['nir'].sel(location=idx, time=planting_slice) / 10000
    red = ds['red'].sel(location=idx, time=planting_slice) / 10000
    ndvi = (nir - red) / (nir + red + .00001)

    #start_indx = ndvi.idxmin(dim=time).values
    #end_indx = start_indx + np.timedelta64(1, 'Y').astype('timedelta64[ns]')

    coastal = ds['coastal'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    blue = ds['blue'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    green = ds['green'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    red = ds['red'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    veg5 = ds['veg5'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    veg6 = ds['veg6'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    veg7 = ds['veg7'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    nir = ds['nir'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    narrow_nir = ds['narrow_nir'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    water_vapour = ds['water_vapour'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    swir1 = ds['swir1'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    swir2 = ds['swir2'].sel(location=idx, time=slice(start_indx, end_indx)) / 10000
    SCL = ds['SCL'].sel(location=idx, time=slice(start_indx, end_indx))
    WVP = ds['WVP'].sel(location=idx, time=slice(start_indx, end_indx)) / 1000
    AOT = ds['AOT'].sel(location=idx, time=slice(start_indx, end_indx)) / 1000

    ndvi = (nir - red) / (nir + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)
    ndbi = (swir1 - red) / (swir1 + red + .00001)

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

    return (
    tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ndvi.time.values)


def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def loops_fill(arr):
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(out.shape[1]):
            for t_idx in range(1, out.shape[2]):
                if np.isnan(out[row_idx, col_idx, t_idx]):
                    out[row_idx, col_idx, t_idx] = out[row_idx, col_idx, t_idx - 1]
    return out


@click.command()
@click.option('--crop', '-c', type=click.Choice(['SUGARCANE', 'RICE', 'ALL'], case_sensitive=False), default='ALL',
              help='create model for which crop?', show_default=True)
@click.option('--seed', '-s', default=1, show_default=True, help='seed number for random generator')
@click.option('--output', '-o', default='', type=click.Path(exists=True, file_okay=False, dir_okay=True, ),
              help='output directory for model')
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=False, dir_okay=True, ),
              help='input file')
@click.option('--n_samp', '-n', default=10000, show_default=True, help='number of samples for each class')
def start(crop, seed, output, input, n_samp):
    print("Start shifting")
    print("n_samp:", n_samp)
    print("crop:", crop)

    np.random.seed(seed)

    # data = 'c:/flk_repository/dataset/trainSet3classRMMEH.npz'

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

    #for file in data:
    for i in range(0,len(data)):
        file = data[i]

        print('read file:', input + '/' + file)
        ds = xr.open_dataset(input + '/' + file)

        band = list(ds.keys())
        band.append('ndvi')
        band.append('ndwi')

        y = ds.y.values

        n_classes = None
        other_idx_trn = None
        target_idx_trn = None

        other_idx_test = None
        target_idx_test = None

        planting_slice = None

        if crop == 'ALL':
            n_classes = 3
            # MODEL_PATH = "./multilayer_model/narlab_3cls_rmmeh.pth"
            # idx = np.concatenate([other_idx, sugar_idx, rice_idx])
            pass
        elif crop == 'SUGARCANE':
            n_classes = 2
            # MODEL_PATH = "./multilayer_model/narlab_sugar_rmmeh.pth"
            other_idx_trn = np.random.choice(np.where(y != 1)[0], n_samp)
            target_idx_trn = np.random.choice(np.where(y == 1)[0], n_samp)

            other_idx_test = np.random.choice(np.where(y != 1)[0], n_samp)
            target_idx_test = np.random.choice(np.where(y == 1)[0], n_samp)

            planting_slice = date_slice_sugar[i]
        elif crop == 'RICE':
            n_classes = 2
            # MODEL_PATH = "./multilayer_model/narlab_rice_rmmeh.pth"

            other_idx_trn = np.random.choice(np.where(y != 2)[0], n_samp)
            target_idx_trn = np.random.choice(np.where(y == 2)[0], n_samp)

            other_idx_test = np.random.choice(np.where(y != 2)[0], n_samp)
            target_idx_test = np.random.choice(np.where(y == 2)[0], n_samp)

            planting_slice = date_slice_rice[i]

        trn_idx = np.concatenate([other_idx_trn, target_idx_trn])
        test_idx = np.concatenate([other_idx_test, target_idx_test])

        (tmp_array, y, position, location, timestamp) = sampling_ds(ds, trn_idx, planting_slice)

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

        (tmp_array, y, position, location, timestamp) = sampling_ds(ds, test_idx, planting_slice)

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
    print("Shape of timestamp:", trn_time.shape)



    # Save to NPY format
    np.savez(output + "/trainSet_"+crop+"_RMMEH_masked.npz", x=trn_x, y=trn_y, location=trn_location, position=trn_pos,
             timestamp=trn_time, bands=np.array(band))
    np.savez(output + "/testSet_"+crop+"_RMMEH_masked.npz", x=test_x, y=test_y, location=test_location, position=test_pos,
             timestamp=test_time, bands=np.array(band))

if __name__ == '__main__':
    start()

'''
    npzfile = np.load(data, allow_pickle=True)

    x_trn = npzfile['x'].astype(np.float32)
    y_trn = npzfile['y'].astype(np.float32)

    other_idx = np.random.choice(np.where(y_trn == 0)[0], n_samp)
    sugar_idx = np.random.choice(np.where(y_trn == 1)[0], n_samp)
    rice_idx = np.random.choice(np.where(y_trn == 2)[0], n_samp)

    idx = None
    if crop == 'ALL':
        n_classes = 3
        #MODEL_PATH = "./multilayer_model/narlab_3cls_rmmeh.pth"
        idx = np.concatenate([other_idx, sugar_idx, rice_idx])
    elif crop == 'SUGARCANE':
        n_classes = 2
        #MODEL_PATH = "./multilayer_model/narlab_sugar_rmmeh.pth"
        idx = np.concatenate([other_idx, sugar_idx])
    elif crop == 'RICE':
        n_classes = 2
        #MODEL_PATH = "./multilayer_model/narlab_rice_rmmeh.pth"
        idx = np.concatenate([other_idx, rice_idx])

    x_trn = np.take(x_trn, idx, axis=0)
    y_trn = np.take(y_trn, idx)

    A = np.any(np.isnan(x_trn))
    #print(A)

    series_length = x_trn.shape[2]
    band = npzfile['bands'].tolist()
    # band = ['narrow_nir','ndvi','WVP', 'SCL']
    n_features = len(band)

    arr = np.arange(x_trn.shape[0])
    np.random.shuffle(arr)

    x_trn = np.take(x_trn, arr, axis=0)
    y_trn = np.take(y_trn, arr)
'''
