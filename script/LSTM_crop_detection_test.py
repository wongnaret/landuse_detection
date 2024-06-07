#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: crop_detection_test.py
 Date Create: 17/11/2021 AD 02:47
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

from datetime import datetime

import click
import numpy as np
import pandas as pd
import pytz
import rasterio
import torch
import xarray as xr
from rasterio.transform import Affine

from LSTMClassifier import LSTMClassifier
from accuracy import export_confusion_matrix
from lstm_utils import loops_fill, numpy_fill, convert_to_df, feed_confident, feed, feed_raw

from scipy import ndimage, misc
from scipy import stats

seed = 999
NUM_LAYERS = 8

#n_classes = 3

# bs = 128
bs = 256

# h = [256]
h = [128]

# lr = 0.0005
lr = 0.001
import copy


def original(i, j, img, ksize=5):
    # Find the matrix coordinates
    x1 = y1 = -ksize // 2
    x2 = y2 = ksize + x1
    temp = np.zeros(ksize * ksize)
    count = 0
    # Process images
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j]
            else:
                temp[count] = img[i + m, j + n]
            count += 1
    return temp


def max_vote_function(img, ksize=5):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            temp = original(i, j, img0, ksize).astype(np.int32)
            img[i, j] = np.max(np.bincount(temp.flatten()).argmax())

            # if flag == 0:   # set the flag parameter to detect the maximum value if 0 and the minimum value if 1.
            #    img[i, j, k] = np.max(temp)
            # elif flag == 1:
            #    img[i, j, k] = np.min(temp)
    return img

def my_linear(x):
    return x*(-0.374)+1.823

def sampling_ds(ds, idx, time_slice):
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
    ndwi = (green - nir) / (green + nir + .00001)
    '''

    if idx is None:
        nir = ds['nir'].sel(time=time_slice) / 10000
        red = ds['red'].sel(time=time_slice) / 10000
        green = ds['green'].sel(time=time_slice) / 10000
        swir1 = ds['swir1'].sel(time=time_slice) / 10000
        scl = ds['SCL'].sel(time=time_slice)

        y = ds.y
        y[y == 3] = 0

        i = ds.row
        j = ds.col

        lat = ds.latitude
        lon = ds.longitude

    else:
        nir = ds['nir'].sel(location=idx, time=time_slice) / 10000
        red = ds['red'].sel(location=idx, time=time_slice) / 10000
        green = ds['green'].sel(time=time_slice) / 10000
        swir1 = ds['swir1'].sel(location=idx, time=time_slice) / 10000
        scl = ds['SCL'].sel(location=idx, time=time_slice)

        y = ds.y.sel(location=idx)
        y[y == 3] = 0

        i = ds.row.sel(location=idx)
        j = ds.col.sel(location=idx)

        lat = ds.latitude.sel(location=idx)
        lon = ds.longitude.sel(location=idx)

    ndvi = (nir - red) / (nir + red + .00001)
    ndbi = (swir1 - red) / (swir1 + red + .00001)
    ndwi = (green - nir) / (green + nir + .00001)

    tmp_array = np.array([ndvi, ndbi, ndwi])
    position = np.array([i, j])

    location = np.array([lat, lon])

    return (tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ndvi.time.values)


def sampling_ds_classes(ds, idx, time_slice, classes, is_veg=False):
    if idx is None:
        nir = ds['nir'].sel(time=time_slice) / 10000
        red = ds['red'].sel(time=time_slice) / 10000
        green = ds['green'].sel(time=time_slice) / 10000
        swir1 = ds['swir1'].sel(time=time_slice) / 10000
        scl = ds['SCL'].sel(time=time_slice)

        y = ds.y
        #y[y == 3] = 0
        for i in range(0,5):
            #print("Iter:",i)
            #print("Classes",classes)
            if str(i) not in classes: 
                y[y == i] = 0

        if is_veg:
            print("Trainnign cor Vegetation detection")
            y[y == 2] = 1
            y[y == 3] = 1

        i = ds.row
        j = ds.col

        lat = ds.latitude
        lon = ds.longitude

    else:    
        nir = ds['nir'].sel(location=idx, time=time_slice) / 10000
        red = ds['red'].sel(location=idx, time=time_slice) / 10000
        green = ds['green'].sel(location=idx, time=time_slice) / 10000
        swir1 = ds['swir1'].sel(location=idx, time=time_slice) / 10000

        scl = ds['SCL'].sel(location=idx, time=time_slice) 

        y = ds.y.sel(location=idx)
        for i in range(0,5):
            #print("Iter:",i)
            #print("Classes",classes)
            if str(i) not in classes: 
                y[y == i] = 0

        if is_veg:
            print("Trainnign cor Vegetation detection")
            y[y == 2] = 1
            y[y == 3] = 1

        i = ds.row.sel(location=idx)
        j = ds.col.sel(location=idx)
        
        lat = ds.latitude.sel(location=idx)
        lon = ds.longitude.sel(location=idx)
    

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

   
        
        
    tmp_array = np.array([ndvi, ndbi, ndwi])
    position = np.array([i, j])
    location = np.array([lat, lon])

    unique, counts = np.unique(y.values, return_counts=True)

    print("prepareing data", np.asarray((unique, counts)).T)

    return (
        tmp_array.transpose((1, 0, 2)), y.values, position.transpose(), location.transpose(), ndvi.time.values)



@click.command()
@click.option('--gpu/--cpu', default=False, help=' select cpu or gpu', show_default=True)
@click.option('--veg/--crop', default=False, help=' select cpu or gpu', show_default=True)
@click.option('--output', '-o', default='', type=click.Path(exists=False, file_okay=True, dir_okay=False, ),
              help='output directory for prediction results')
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='input NC file')
@click.option('--model_path', '-m', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='input LSTM model')
@click.option('--n_samp', '-n', default=-1, show_default=True, help='number of samples for each class')
@click.option('--classes', '-c', multiple=True, default=[0])
@click.option('--groundtruth', '-g', default=None, type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='groundtruth file (*.tif')
def start(gpu, veg, output, input, model_path, n_samp, classes, groundtruth):
    now = datetime.now(pytz.timezone('Asia/Bangkok'))  # current date and time

    timestamp_str = now.strftime("%Y%m%d_%H%M")

    test_x = None
    test_y = None
    test_pos = None
    test_location = None
    test_time = None

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

    if groundtruth is not None:
        groundtruth_tif = rasterio.open(groundtruth)
        groundtruth_arr = groundtruth_tif.read(1)
        print("Groundtruth.shape", groundtruth_arr.shape)
        print("y.shape", y.shape)
        
        groundtruth_arr = groundtruth_arr[::-1, :]
        groundtruth_arr = groundtruth_arr.flatten()
        y = groundtruth_arr
        
    #n_classes = 4
    #n_classes = 3
    n_classes = len(classes)

    other_idx_trn = None
    target_idx_trn = None

    other_idx_test = None
    target_idx_test = None

    #planting_slice = slice('2018-11-01', '2019-05-30')
    planting_slice = slice('2021-04-01', '2022-04-30')

    test_idx = None

    if n_samp > 0:
        #other_idx_test = np.random.choice(np.where(y != 1)[0], n_samp)
        #target_idx_test = np.random.choice(np.where(y == 1)[0], n_samp)

        #test_idx = np.concatenate([other_idx_test, target_idx_test])
        # MODEL_PATH = "./multilayer_model/narlab_sugar_rmmeh.pth"

        other_idx_test = np.random.choice(np.where(y == 0)[0], n_samp)
        sugar_idx_test = np.random.choice(np.where(y == 1)[0], n_samp)
        rice_idx_test = np.random.choice(np.where(y == 2)[0], n_samp)
        veg_idx_test = np.random.choice(np.where(y ==3)[0], n_samp)

        test_idx = np.concatenate([other_idx_test, sugar_idx_test, rice_idx_test, veg_idx_test])

    #(tmp_array, y, position, location, timestamp) = sampling_ds(ds, test_idx, planting_slice)
    (tmp_array, y, position, location, timestamp) = sampling_ds_classes(ds, test_idx, planting_slice, classes, veg)

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

    print("Shape of x:", test_x.shape)
    print("Shape of y:", test_y.shape)
    print("Shape of position:", test_pos.shape)
    print("Shape of location:", test_location.shape)
    print("Shape of timestamp:", test_time.shape)

    x_trn = test_x
    y_trn = test_y

    series_length = x_trn.shape[2]

    n_features = len(band)

    A = np.any(np.isnan(x_trn))
    print(A)
    print("Nan fill")
    
    x_trn = loops_fill(x_trn)

    print("Nan to num fill")

    for i in range(x_trn.shape[2]):
        x_trn[:, :, i] = numpy_fill(x_trn[:, :, i])
    x_trn = np.nan_to_num(x_trn, nan=0.0001)

    print("Create dataframe")

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

    print("Load LSTM model")
    # Training Loop
    # Howmany feature
    input_dim = n_features

    # Howmany class
    # output_dim = 2
    output_dim = n_classes

    hidden_dim = h[0]
    layer_dim = NUM_LAYERS

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, gpu)

    if gpu:
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.cpu()

    model.eval()

    print('Feeding data to model...')
    #(y_pred, full_result) = feed_confident(x_trn, model, gpu)
    (y_pred, full_result) = feed_raw(x_trn, model, gpu)
    
    y_actu = y_trn['class'].tolist()


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
    y_pred = np.array(y_pred)
    y_actu = np.array(y_actu)

    y_pred_img = y_pred.reshape((ds.col.max() + 1).item(0), (ds.row.max() + 1).item(0))
    y_pred_img = y_pred_img.astype(np.int32)
    with rasterio.open(
            output,
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

    
    y_pred_img = ndimage.median_filter(y_pred_img, size=5)

    #export_confusion_matrix(y_actu.flatten(), y_pred_img.flatten(), "%s/confusion_matrix_median_%s.png" % (output, timestamp_str))

    with rasterio.open(
            output.replace(".tif", '_median.tif'),
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

    conf_lv = np.array(full_result)

    conf_max = conf_lv.max(axis=1)
    conf_max_img = conf_max.reshape((ds.col.max() + 1).item(0), (ds.row.max() + 1).item(0))

    '''
    with rasterio.open(
            output + '/conf_lv_by_LSTM_%s.tif'%(timestamp_str),
            'w',
            driver='GTiff',
            height=conf_max_img.shape[0],
            width=conf_max_img.shape[1],
            count=1,
            dtype=conf_max_img.dtype,
            crs='+proj=latlong',
            transform=transform,
    ) as dst:
        dst.write(conf_max_img[::-1, :], 1)
    '''

    # display accuracy and confusion matrix
    print("Exporting result and accracy matrix")
    #export_confusion_matrix(y_actu, y_pred, "%s/confusion_matrix_%s.png" % (output, timestamp_str))

    # display accuracy and confusion matrix
    #export_confusion_matrix(y_actu, y_pred, "%s/confusion_matrix_%s.png" % (output, timestamp_str))

    print('full_result')

    conf_lv = np.array(full_result)
    print(conf_lv)

    #np.save("%s/conf_lv_%s.npy" % (output, timestamp_str), conf_lv)
    #np.save("%s/groundtruth_%s.npy" % (output, timestamp_str), y_actu)
    #np.save("%s/prediction_%s.npy" % (output, timestamp_str), y_pred)

    ratio = 10

    df = pd.DataFrame(conf_lv)
    df = df.rename(columns={0: "other", 1: "rice", 2: "sugarcane"})

    sorted_df = df.sort_values('other')
    sorted_df = sorted_df.reset_index()
    sorted_df = sorted_df[['other', 'rice', 'sugarcane']]
    tmp = np.array((sorted_df['other'], sorted_df['rice'], sorted_df['sugarcane']))
    tmp.mean(axis=0)
    # sorted_df['dist'] = abs(sorted_df['other']-sorted_df['rice'],)
    sorted_df['dist'] = np.linalg.norm(tmp)

    df['dist'] = np.linalg.norm(np.array((df['other'], df['rice'], df['sugarcane'])))
    df['dist2'] = abs(np.array((df['other'], df['rice'], df['sugarcane'])).mean(axis=0))

    #min_dist = df[df['dist2'] == df['dist2'].min()]
    #pr = stats.percentileofscore(sorted_df['other'].values, min_dist.values)

    df['ground_truth'] = y_actu
    df['lstm_predict'] = y_pred

    lower_treshold = df['dist2'].quantile((ratio) / 100)

    df['test'] = df['lstm_predict']
    df['test'][df['dist2'] <= lower_treshold] = 3

    test_img = df['test'].values.reshape((ds.col.max() + 1).item(0), (ds.row.max() + 1).item(0))
    test_img = test_img.astype(np.int32)

    with rasterio.open(
            output.replace(".tif", "_with_conf.tif"),
            'w',
            driver='GTiff',
            height=test_img.shape[0],
            width=test_img.shape[1],
            count=1,
            dtype=test_img.dtype,
            crs='+proj=latlong',
            transform=transform,

    ) as dst:
        dst.meta['nodata'] = -9999
        dst.write(test_img[::-1, :], 1)

    test_img = max_vote_function(test_img)

    with rasterio.open(
            output.replace(".tif", "_with_conf_noise_reduced.tif"),
            'w',
            driver='GTiff',
            height=test_img.shape[0],
            width=test_img.shape[1],
            count=1,
            dtype=test_img.dtype,
            crs='+proj=latlong',
            transform=transform,

    ) as dst:
        dst.meta['nodata'] = -9999
        dst.write(test_img[::-1, :], 1)


if __name__ == '__main__':
    start()
