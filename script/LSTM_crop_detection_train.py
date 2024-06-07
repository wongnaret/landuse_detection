#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: crop_detection_LSTM.py
 Date Create: 16/11/2021 AD 00:11
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

# roc curve for logistic regression model with optimal threshold

from datetime import datetime

import click
import numpy as np
import pandas as pd
import pytz
import torch
import xarray as xr
from torch import nn
from torch.nn import functional as F
import rasterio
from rasterio.transform import Affine

from LSTMClassifier import LSTMClassifier
from lstm_utils import loops_fill, numpy_fill, convert_to_df, create_datasets, create_loaders, CyclicLR, cosine

from scipy import ndimage, misc

seed = 999
NUM_LAYERS = 8

MODEL_PATH = None

# bs = 128
bs = 256

# h = [256]
h = [128]

# lr = 0.0005
lr = 0.001


def sampling_ds(ds, idx, time_slice):
    nir = ds['nir'].sel(location=idx, time=time_slice) / 10000
    red = ds['red'].sel(location=idx, time=time_slice) / 10000
    green = ds['green'].sel(location=idx, time=time_slice) / 10000
    swir1 = ds['swir1'].sel(location=idx, time=time_slice) / 10000

    scl = ds['SCL'].sel(location=idx, time=time_slice)

    ndvi = (nir - red) / (nir + red + .00001)
    ndbi = (swir1 - red) / (swir1 + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)

    # start_indx = ndvi.idxmin(dim=time).values
    # end_indx = start_indx + np.timedelta64(1, 'Y').astype('timedelta64[ns]')

    '''
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
    
    '''

    y = ds.y.sel(location=idx)
    y[y == 3] = 0

    tmp_array = np.array([ndvi, ndbi, ndwi])

    i = ds.row.sel(location=idx)
    j = ds.col.sel(location=idx)
    position = np.array([i, j])

    lat = ds.latitude.sel(location=idx)
    lon = ds.longitude.sel(location=idx)
    location = np.array([lat, lon])

    return (
        tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ndvi.time.values)

def sampling_ds_classes(ds, idx, time_slice, classes, is_veg=False):
    nir = ds['nir'].sel(location=idx, time=time_slice) / 10000.0
    red = ds['red'].sel(location=idx, time=time_slice) / 10000.0
    green = ds['green'].sel(location=idx, time=time_slice) / 10000.0
    swir1 = ds['swir1'].sel(location=idx, time=time_slice) / 10000.0

    scl = ds['SCL'].sel(location=idx, time=time_slice) 

    ndvi = (nir - red) / (nir + red + .00001)
    ndbi = (swir1 - red) / (swir1 + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)

    # start_indx = ndvi.idxmin(dim=time).values
    # end_indx = start_indx + np.timedelta64(1, 'Y').astype('timedelta64[ns]')

    '''
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
    
    '''

    y = ds.y.sel(location=idx)
    for i in range(0,5):
        #print("Iter:",i)
        #print("Classes",classes)
        if str(i) not in classes: 
            y[y == i] = 0

    if is_veg:
        print("Training for Vegetation detection")
        y[y == 2] = 1
        y[y == 3] = 1
        
        
    tmp_array = np.array([ndvi, ndbi, ndwi])

    i = ds.row.sel(location=idx)
    j = ds.col.sel(location=idx)
    position = np.array([i, j])

    lat = ds.latitude.sel(location=idx)
    lon = ds.longitude.sel(location=idx)
    location = np.array([lat, lon])

    unique, counts = np.unique(y.values, return_counts=True)

    print("prepareing data", np.asarray((unique, counts)).T)

    return (
        tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ndvi.time.values)


def prepare_x(x_trn, band):
    series_length = x_trn.shape[2]


    trn = {}

    for tmp in band:
        trn[tmp] = x_trn[:,band.index(tmp),:]
        print(tmp, 'Extraction shape:', trn[tmp].shape)
    n_features = len(trn)

    for product in trn:
        print("Product", product)
        trn[product] = convert_to_df(trn[product], product)
        print("dim", trn[product].shape)

    trn_x = trn['ndvi'].merge(trn['ndwi']['ndwi'], how='left',
                              left_index=True,
                              right_index=True)
    for product in trn:
        if product in ['ndvi', 'ndwi']:
            continue

        trn_x = trn_x.merge(trn[product][product], how='left',
                            left_index=True,
                            right_index=True)

    # x_trn['measurement_number'].unique()
    return (trn_x, series_length, n_features)

def prepare_y(y_trn, all_rows=False):
    y_trn = pd.DataFrame(y_trn)
    y_trn = y_trn.reset_index()
    y_trn = y_trn.rename(columns={"index": "series_id", 0: "class"}, errors="raise")

    n_classes = y_trn['class'].nunique()

    # y_trn = y_trn.reset_index()
    # y_trn = y_trn.drop(columns=['index'])
    return (y_trn, n_classes)

def export_raster(filename, y, ds):
    # export to GEO-Tiff image
    #  Initialize the Image Size
    image_size = ((ds.row.max() + 1).item(0), (ds.col.max() + 1).item(0))

    #  Choose some Geographic Transform (Around Lake Tahoe)
    lat = [ds['latitude'].min().item(), ds['latitude'].max().item()]
    lon = [ds['longitude'].min().item(), ds['longitude'].max().item()]

    
    # set geotransform
    nx = image_size[0]
    ny = image_size[1]
    xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    transform = Affine.translation(xmin - xres / 2, ymin - yres / 2) * Affine.scale(xres, yres)

    # export raster
    y_pred = np.array(y)

    y_pred_img = y_pred.reshape((ds.col.max() + 1).item(0), (ds.row.max() + 1).item(0))
    y_pred_img = y_pred_img.astype(np.int32)
    with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=y_pred_img.shape[0],
            width=y_pred_img.shape[1],
            count=1,
            dtype=y_pred_img.dtype,
            crs='+proj=latlong',
            transform=transform,

    ) as dst:
        dst.write(y_pred_img[::-1, :], 1)

@click.command()
@click.option('--gpu/--cpu', default=False, help=' select cpu or gpu', show_default=True)
@click.option('--veg/--crop', default=False, help=' select vegetation detection or crop detection', show_default=True)
@click.option('--output', '-o', default='', type=click.Path(exists=True, file_okay=False, dir_okay=True, ),
              help='output directory for model')
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='input NC file')
@click.option('--nc/--npz', default=True, help=' select input file type', show_default=True)
@click.option('--n_samp', '-n', default=20000, show_default=True, help='number of samples for each class')
@click.option('--epoch', '-e', default=20, show_default=True, help='maximum epoch for training')
@click.option('--classes', '-c', multiple=True, default=[0])
@click.option('--groundtruth', '-g', default=None, type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='groundtruth file (*.tif')
@click.option('--mask', default=None, type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='mask file (*.tif')
@click.option('--test/--run', default=False, help=' select test for exporting groundtruth', show_default=True)

def start(gpu, veg, output, input, nc, n_samp, epoch, classes, groundtruth, mask, test):
    if nc:
        start_nc(gpu, veg, output, input, n_samp, epoch, classes, groundtruth, mask, test)
    else :
        start_npz(gpu, veg, output, input, n_samp, epoch, classes, groundtruth, mask, test)

def start_npz(gpu, veg, output, input, n_samp,  epoch, classes, groundtruth, mask, test):
    now = datetime.now(pytz.timezone('Asia/Bangkok'))  # current date and time

    timestamp_str = now.strftime("%Y%m%d_%H%M")

    x_trn = None
    y_trn = None
    trn_pos = None
    trn_location = None
    trn_time = None

    print("Start training")
    print("is gpu:", gpu)

    np.random.seed(seed)

    if gpu:
        torch.cuda.set_device(0)

    # data = 'c:/flk_repository/dataset/trainSet3classRMMEH.npz'
    data = input

    print('read file:', data)

    npzfile = np.load(input, allow_pickle=True)

    #ds = xr.open_dataset(data)

    #band = npzfile['bands'].tolist()
    band = ['vh','vv']
    #band.append('ndvi')
    #band.append('ndbi')
    #band.append('ndwi')

    print("Original shape of x:", npzfile['x'].shape)
    print("Original shape of y:", npzfile['y'].shape)

    trn_x = npzfile['x'].reshape(-1, npzfile['x'].shape[2], npzfile['x'].shape[3])
    trn_y = npzfile['y'].reshape(-1)


    print("Reshaped of x:", trn_x.shape)
    print("Reshaped of y:", trn_y.shape)


    A = np.any(np.isnan(trn_x))
    print("Is training set has nan", A)

    print("Starting loop-fill")
    trn_x = loops_fill(trn_x)

    print("Starting nan-to-num")
    for i in range(trn_x.shape[2]):
        trn_x[:, :, i] = numpy_fill(trn_x[:, :, i])
    trn_x = np.nan_to_num(trn_x, nan=0.0001)

    series_length = trn_x.shape[2]

    # band = ['narrow_nir','ndvi','WVP', 'SCL']
    n_features = len(band)

    arr = np.arange(trn_x.shape[0])
    np.random.shuffle(arr)

    trn_x = np.take(trn_x, arr, axis=0)
    trn_y = np.take(trn_y, arr)


    (x_trn, series_length, n_features) = prepare_x(trn_x, band)
    (y_trn, n_classes) = prepare_y(trn_y)

    print("Shape of x:", x_trn.shape)
    print("Shape of y:", y_trn.shape)
    print('Preparing datasets')
    
    trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['class'])

    print(f'Creating data loaders with batch size: {bs}')
    # trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=cpu_count())
    trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=1)

    result_acc = []
    stop_epoch = []

    for i in range(0, len(h)):
        # Training Loop
        layer_dim = NUM_LAYERS
        # hidden_dim = 256

        hidden_dim = h[i]

        # Howmany feature
        input_dim = n_features

        # Howmany class
        # output_dim = 3
        output_dim = n_classes

        seq_dim = series_length

        iterations_per_epoch = len(trn_dl)
        best_acc = 0
        patience, trials = 100, 0

        print("Hidden layer size =", hidden_dim, "Iterations/epoch =", iterations_per_epoch, "Max epoch =", epoch)

        model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, gpu)
        if gpu:
            model = model.cuda()
        else:
            model = model.cpu()

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

        # opt = torch.optim.SGD(model.parameters(), lr=0.1)
        # sched = StepLR(opt, step_size=4, gamma=0.1)

        print('Start model training')

        for i_epoch in range(1, epoch + 1):

            for i, (x_batch, y_batch) in enumerate(trn_dl):
                model.train()
                if gpu:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                else:
                    x_batch = x_batch.cpu()
                    y_batch = y_batch.cpu()

                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                # sched.step()
                opt.step()
                sched.step()

            model.eval()
            correct, total = 0, 0
            for x_val, y_val in val_dl:
                if gpu:
                    x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                else:
                    x_val, y_val = [t.cpu() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                total += y_val.size(0)
                correct += (preds == y_val).sum().item()

            acc = correct / total

            print(f'Epoch: {i_epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

            if acc > best_acc:
                trials = 0
                best_acc = acc
                # torch.save(model.state_dict(), MODEL_PATH)
                torch.save(model.state_dict(), output + '/crop_detection_' + timestamp_str + '_rmmeh.pth')
                print(f'Epoch {i_epoch} best model saved with accuracy: {best_acc:2.2%}')
                if (i < len(result_acc)):
                    result_acc[i] = best_acc
                    stop_epoch[i] = epoch
                else:
                    result_acc.append(best_acc)
                    stop_epoch.append(epoch)
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {i_epoch}')
                    break

    #torch.save(model.state_dict(), output + '/crop_detection_' + timestamp_str + '_rmmeh_last_epoch.pth')

def start_nc(gpu, veg, output, input, n_samp, epoch, classes, groundtruth, mask, test):
    now = datetime.now(pytz.timezone('Asia/Bangkok'))  # current date and time

    timestamp_str = now.strftime("%Y%m%d_%H%M")

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

    print("Start training")
    print("is gpu:", gpu)
    print("Target classes:", classes)

    np.random.seed(seed)

    if gpu:
        torch.cuda.set_device(0)

    # data = 'c:/flk_repository/dataset/trainSet3classRMMEH.npz'
    data = input

    print('read file:', data)
    ds = xr.open_dataset(data)

    # band = list(ds.keys())
    band = []
    band.append('ndvi')
    band.append('ndbi')
    band.append('ndwi')

    y = ds.y.values

    if test:
        export_raster(output+'/original_ground_truth_%s.tif'%(timestamp_str), y, ds)
        exit()

    if mask is not None:
        mask_tif = rasterio.open(mask)
        mask_arr = mask_tif.read(1)
        print("Mask.shape", mask_arr.shape)

        mask_arr = mask_arr[::-1, :]
        mask_arr = mask_arr.flatten()

    if groundtruth is not None:
        groundtruth_tif = rasterio.open(groundtruth)
        groundtruth_arr = groundtruth_tif.read(1)
        print("Groundtruth.shape", groundtruth_arr.shape)
        print("y.shape", y.shape)
        
        groundtruth_arr = groundtruth_arr[::-1, :]
        groundtruth_arr = groundtruth_arr.flatten()
        y = groundtruth_arr

    n_classes = len(classes)

    print("sampling training data.")

    other_idx_trn = None
    target_idx_trn = None

    other_idx_test = None
    target_idx_test = None

    #planting_slice = slice('2018-11-01', '2019-05-30')
    planting_slice = slice('2021-04-01', '2022-04-30')

    # MODEL_PATH = "./multilayer_model/narlab_sugar_rmmeh.pth"

    factor = 5-n_classes
    if mask is not None:
        if veg:
            other_idx_trn = np.random.choice(np.where((y == 0) & (mask_arr == 1))[0], int(n_samp))
            other_idx_test = np.random.choice(np.where((y == 0) & (mask_arr == 1))[0], int(n_samp))
        else:
            other_idx_trn = np.random.choice(np.where((y == 0) & (mask_arr == 1))[0], int(n_samp))
            other_idx_test = np.random.choice(np.where((y == 0) & (mask_arr == 1))[0], int(n_samp))

        if '1' in classes:
            if veg:
                sugar_idx_trn = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], int(n_samp / 3))
                sugar_idx_test = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], int(n_samp / 3))
            else:
                sugar_idx_trn = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], n_samp)
                sugar_idx_test = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], n_samp)
        else:
            sugar_idx_trn = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], int(n_samp / factor))
            sugar_idx_test = np.random.choice(np.where((y == 1) & (mask_arr == 1))[0], int(n_samp / factor))

        if '2' in classes:
            if veg:
                rice_idx_trn = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], int(n_samp / 3))
                rice_idx_test = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], int(n_samp / 3))
            else:
                rice_idx_trn = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], n_samp)
                rice_idx_test = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], n_samp)
        else:
            rice_idx_trn = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], int(n_samp / factor))
            rice_idx_test = np.random.choice(np.where((y == 2) & (mask_arr == 1))[0], int(n_samp / factor))


    else:
        if veg:
            other_idx_trn = np.random.choice(np.where(y == 0)[0], int(n_samp))
            other_idx_test = np.random.choice(np.where(y == 0)[0], int(n_samp))
        else:
            other_idx_trn = np.random.choice(np.where(y == 0)[0], int(n_samp))
            other_idx_test = np.random.choice(np.where(y == 0)[0], int(n_samp))

        if '1' in classes:
            if veg:
                sugar_idx_trn = np.random.choice(np.where(y == 1)[0], int(n_samp/3))
                sugar_idx_test = np.random.choice(np.where(y == 1)[0], int(n_samp/3))
            else:
                sugar_idx_trn = np.random.choice(np.where(y == 1)[0], n_samp)
                sugar_idx_test = np.random.choice(np.where(y == 1)[0], n_samp)
        else:
            sugar_idx_trn = np.random.choice(np.where(y == 1)[0], int(n_samp/factor))
            sugar_idx_test = np.random.choice(np.where(y == 1)[0], int(n_samp/factor))

        if '2' in classes:
            if veg:
                rice_idx_trn = np.random.choice(np.where(y == 2)[0], int(n_samp/3))
                rice_idx_test = np.random.choice(np.where(y == 2)[0], int(n_samp/3))
            else:
                rice_idx_trn = np.random.choice(np.where(y == 2)[0], n_samp)
                rice_idx_test = np.random.choice(np.where(y == 2)[0], n_samp)
        else:
            #rice_idx_trn = np.random.choice(np.where(y == 2)[0], int(n_samp/factor))
            #rice_idx_test = np.random.choice(np.where(y == 2)[0], int(n_samp/factor))
            rice_idx_trn = []
            rice_idx_test =[]

    #trn_idx = np.concatenate([other_idx_trn, sugar_idx_trn, rice_idx_trn, veg_idx_trn])
    #test_idx = np.concatenate([other_idx_test, sugar_idx_test, rice_idx_test, veg_idx_test])
    trn_idx = np.concatenate([other_idx_trn, sugar_idx_trn, rice_idx_trn])
    test_idx = np.concatenate([other_idx_test, sugar_idx_test, rice_idx_test])

    trn_idx = trn_idx.astype(np.int)
    test_idx = trn_idx.astype(np.int)

    (tmp_array, y, position, location, timestamp) = sampling_ds_classes(ds, trn_idx, planting_slice, classes, veg)

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

    x_trn = trn_x
    y_trn = trn_y

    A = np.any(np.isnan(x_trn))
    print("Is training set has nan", A)

    x_trn = loops_fill(x_trn)

    for i in range(x_trn.shape[2]):
        x_trn[:, :, i] = numpy_fill(x_trn[:, :, i])
    x_trn = np.nan_to_num(x_trn, nan=0.0001)

    series_length = x_trn.shape[2]

    # band = ['narrow_nir','ndvi','WVP', 'SCL']
    n_features = len(band)

    arr = np.arange(x_trn.shape[0])
    np.random.shuffle(arr)

    x_trn = np.take(x_trn, arr, axis=0)
    y_trn = np.take(y_trn, arr)

    trn = {}
    for product in band:
        print("Product", product)
        trn[product] = convert_to_df(x_trn[:, band.index(product), :], product)

    # x_trn = trn['ndvi'].merge(trn['ndwi']['ndwi'], how='left',
    #                                      left_index=True,
    #                                      right_index=True)
    x_trn = trn['ndvi'].merge(trn['ndbi']['ndbi'], how='left',
                              left_index=True,
                              right_index=True)

    for product in trn:
        if product in ['ndvi', 'ndbi']:
            # if product in ['ndvi', 'WVP']:
            continue

        x_trn = x_trn.merge(trn[product][product], how='left',
                            left_index=True,
                            right_index=True)

    y_trn = pd.DataFrame(y_trn)
    y_trn = y_trn.reset_index()
    y_trn = y_trn.rename(columns={"index": "series_id", 0: "class"}, errors="raise")

    print('Preparing datasets')
    trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['class'])

    print(f'Creating data loaders with batch size: {bs}')
    # trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=cpu_count())
    trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=1)

    result_acc = []
    stop_epoch = []

    for i in range(0, len(h)):
        # Training Loop
        layer_dim = NUM_LAYERS
        # hidden_dim = 256

        hidden_dim = h[i]

        # Howmany feature
        input_dim = n_features

        # Howmany class
        # output_dim = 3
        output_dim = n_classes

        seq_dim = series_length

        iterations_per_epoch = len(trn_dl)
        best_acc = 0
        patience, trials = 100, 0

        print("Hidden layer size =", hidden_dim, "Iterations/epoch =", iterations_per_epoch, "Max epoch =", epoch)

        model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, gpu)
        if gpu:
            model = model.cuda()
        else:
            model = model.cpu()

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

        # opt = torch.optim.SGD(model.parameters(), lr=0.1)
        # sched = StepLR(opt, step_size=4, gamma=0.1)

        print('Start model training')

        best_state = None

        start_time = datetime.now()

        for i_epoch in range(1, epoch + 1):

            for i, (x_batch, y_batch) in enumerate(trn_dl):
                model.train()
                if gpu:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                else:
                    x_batch = x_batch.cpu()
                    y_batch = y_batch.cpu()

                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                # sched.step()
                opt.step()
                sched.step()

            model.eval()
            correct, total = 0, 0
            for x_val, y_val in val_dl:
                if gpu:
                    x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                else:
                    x_val, y_val = [t.cpu() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                total += y_val.size(0)
                correct += (preds == y_val).sum().item()

            acc = correct / total

            end_time = datetime.now()

            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000

            print(f'Epoch: {i_epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%} \t exect time {execution_time:.4f}')

            start_time = end_time

            if acc > best_acc:
                trials = 0
                best_acc = acc
                # torch.save(model.state_dict(), MODEL_PATH)
                best_state = model.state_dict()
                #torch.save(model.state_dict(), output + '/crop_detection_' + timestamp_str + '_rmmeh.pth')
                print(f'Epoch {i_epoch} best model saved with accuracy: {best_acc:2.2%}')
                if (i < len(result_acc)):
                    result_acc[i] = best_acc
                    stop_epoch[i] = epoch
                else:
                    result_acc.append(best_acc)
                    stop_epoch.append(epoch)
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {i_epoch}')
                    break

    torch.save(best_state, '%s/crop_detection_%2.2f_%s_rmmeh.pth'%(output, best_acc, timestamp_str))
    #torch.save(model.state_dict(), output + '/crop_detection_' + timestamp_str + '_rmmeh_last_epoch.pth')


if __name__ == '__main__':
    start()
