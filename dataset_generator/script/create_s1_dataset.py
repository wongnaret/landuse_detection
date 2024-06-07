#!/usr/bin/env python

from unittest.mock import sentinel
import click
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import datacube 
import warnings
warnings.filterwarnings("ignore", ".*Class SelectOfScalar will not make use of SQL compilation caching.*")
from pathlib import Path
import datetime
import rasterio.features
import rasterio
from time import time
import json
import os

def rasterise_label(ds, class_shp):
    print(datetime.datetime.now(), 'raterising class labels')
    gdfs = [ gpd.read_file(s) for s in class_shp ]
    out_shape = (ds.sizes['latitude'], ds.sizes['longitude'])
    label_rst = rasterio.features.rasterize(
        (
            (g,ci)
            for ci, gdf in enumerate(gdfs,1)
            for g in gdf['geometry']
        ),
        out_shape = out_shape,
        fill = 0,
        transform = rasterio.transform.from_bounds(
            west = ds.coords['longitude'][0].item(),
            south = ds.coords['latitude'][-1].item(),
            east = ds.coords['longitude'][-1].item(),
            north = ds.coords['latitude'][0].item(),
            width = ds.sizes['longitude'],
            height = ds.sizes['latitude'],
        ),
        dtype = np.uint16,
    )
    ds = ds.assign_coords(
        y = (('latitude','longitude'), label_rst)
    )
    print(datetime.datetime.now(), 'raterising class labels finished')
    return ds

def save_dataset_nc(name, ds, outdir):
    print('Exporting out dir:', outdir)
    path = outdir + '/' + name + 'Set.nc'
    ds.to_netcdf(path)
    print(datetime.datetime.now(), path, 'written')

@click.command()
@click.option('--product-type', '-p', default='sentinel1_sigma0', help='Sentinel1-product_type')
@click.option('--latitude-extent', type= (float, float) , help='Insert latitude extent')
@click.option('--longitude-extent', type= (float, float), help='Insert longtitude extent')
@click.option('--time-extent', '-t', type= (str, str), help='length of time-series in dataset')
@click.option('--output_dir', '-o', default=os.getcwd(), type=click.Path(file_okay=False, dir_okay=True), help='output directory for result image')
@click.option('--output_name', '-n', type=(str), help='Output file name')
@click.option('--classes', '-c', multiple=True, default = [0], help='path of class label shape file')

def main(
    product_type,
    latitude_extent,
    longitude_extent,
    time_extent,
    output_dir,
    output_name,
    classes,
):
    with datacube.Datacube() as dc:
        dataset = dc.load(latitude = latitude_extent,
                                longitude = longitude_extent,
                                platform = "Sentinel-1",
                                time = time_extent,
                                product = product_type,
                                output_crs='EPSG:4326',
                                #resolution=(-10,10),
                                resolution=(-0.0001345269495781675816, 0.0001345269495781675816)
                            )

    # class_shp = [sugarcane_shp, rice_shp, veg_shp] 

    ## reading shapefiles
    dataset = rasterise_label(dataset, classes)

    ## resample to fill missing week
    # dataset = resample_along_time(dataset, ['vv', 'vh'])

    dataset = dataset.assign_coords(
        row = ('longitude', range(len(dataset.coords['longitude']))),
        col = ('latitude', range(len(dataset.coords['latitude']))),
    )
    dataset = dataset.stack({'location':('latitude','longitude')}).transpose('location','time')
    dataset = dataset.reset_index('location')
    del dataset.time.attrs['units']

    print(dataset)
    # path = output_dir + name + 'Set.nc'
    # dataset.reset_index('location')
    # dataset.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'   # random value
    # print(np.unique(dataset.y.values))
    # dataset.time.encoding['units'] = "seconds since 1970-01-01 00:00:00"
    save_dataset_nc(output_name, dataset, output_dir)

if __name__ == "__main__":
    main()