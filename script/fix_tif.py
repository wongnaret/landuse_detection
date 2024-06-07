#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: fix_tif.py
 Date Create: 9/6/2022 AD 11:01
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""
import click
import rasterio
import os
import rioxarray as rxr
import xarray as xr
from rasterio.transform import Affine
import numpy as np

#Add testing code for GIT DEMO

@click.command()
@click.option('--input', '-f', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
              help='input tif file')
#@click.option('--ds', '-d', default='', type=click.Path(exists=True, file_okay=True, dir_okay=False, ),
#              help='input tif file')
@click.option('--output', '-o', default='', type=click.Path(exists=False, file_okay=False, dir_okay=True, ),
              help='output directory for fixed tif results')

def start(input, output):
    print("Input file:", input)
    print("Output file:", output)

    #ds = rxr.open_rasterio(ds)
    #ds = xr.open_dataset(ds)
    #print("Transform")
    #print(ds.rio.transform())
    #print("CRS", ds.rio.crs)

    '''
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
    '''

    input_raster = rasterio.open(input)

    #transform_org = (~input_raster.transform).translation(1.0, 5.0)
    transform_org = input_raster.transform
    print("Transformation matrix")
    print(transform_org)

    print("reverse transform")
    print(~input_raster.transform)

    print("upper left corner:")
    print(input_raster.transform * (0, 0))

    print("lower right corner:")
    print(input_raster.transform * (input_raster.width, input_raster.height))

    x_max = (input_raster.transform * (input_raster.width, input_raster.height))[0]
    x_min = (input_raster.transform * (0, 0))[0]

    y_max = (input_raster.transform * (input_raster.width, input_raster.height))[1]
    y_min = (input_raster.transform * (0, 0))[1]

    x_offset = x_min
    y_offset = y_max
    x_scale = (x_max - x_min) / input_raster.width
    y_scale = (y_min - y_max) / input_raster.height

    transform = Affine.translation(x_offset, y_offset) * Affine.scale(x_scale, y_scale)

    print("Fixed transform")
    print(transform)

    print("upper left corner:")
    print(transform * (0, 0))

    print("lower right corner:")
    print(transform * (input_raster.width, input_raster.height))

    print("CRS:", input_raster.crs)

    img = input_raster.read(1)

    print("counting...")

    (unique, counts) = np.unique(img.flatten(), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    head, tail = os.path.split(input)
    output_path = "%s/%s"%(output, tail.replace(".tif", "_fixed.tif"))

    print("save output to:", output_path)
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=img.shape[0],
            width=img.shape[1],
            count=1,
            dtype=img.dtype,
            crs='+proj=latlong',
            transform=transform,

    ) as dst:
        dst.write(img[::-1, :], 1)


    return locals()


if __name__ == '__main__':
    globals().update(start(standalone_mode = False))