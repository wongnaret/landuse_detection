"""Extract from datacube into dataset for machine learning workflow."""
import logging
_logger = logging.getLogger(__name__)
import sys
from pathlib import Path
import subprocess
import datetime
# 3rd party modules
import click
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio.features
import rasterio
import xarray as xr
import matplotlib.pyplot as plt
import dask
import datacube
# custom modules
sys.path.insert(0,str(Path(__file__).parent.joinpath('../rmmeh')))
from nd_rmmeh import nd_rmmeh
from kea.clickpath import ClickPath


__version__ = '2.1.2'


@click.group(
    invoke_without_command = True,
    help = __doc__,
)
@click.option(
    '--product',
    default = 'sentinel2_ingestion',
    show_default = True,
    help = 'Name of datacube product to generate dataset for.',
)
@click.option(
    '--latitude-extent',
    type = (float,float),
    default = (16.5634, 16.5354),
    show_default = True,
)
@click.option(
    '--longitude-extent',
    type = (float,float),
    default = (99.6901, 99.7251),
    show_default = True,
)
@click.option(
    '--time-extent',
    type = (str,str),
    default = ('2019-05-01', '2021-05-31'),
    show_default = True,
)
@click.option(
    '--band',
    multiple = True,
    help = 'Name of bands to compile into the dataset. Can be given multiple times. '
        'Do not need to be given again if a band is already given in --inverted-band or --rmmeh-ignore-band.'
        '[default: all bands]',
)
@click.option(
    '--inverted-band',
    multiple = True,
    default = ['coastal','blue','green','red','veg5','veg6','veg7','nir','narrow_nir','water_vapour','swir1','swir2'],
    show_default = True,
    help = 'Mark bands as having abnormality when values are high instead. '
        'This affects how value are selected when groupped by week and when performing RMMEH.'
        'Will automatically be included in --band to be loaded once given.',
)
@click.option(
    '--rmmeh-ignore-band',
    multiple = True,
    default = ['SCL','WVP','AOT'],
    show_default = True,
    help = 'Mark bands to be ignored by RMMEH smoothing. '
        'Will automatically be included in --band to be loaded once given.',
)
@click.option(
    '--n-sample',
    type = int,
    default = 10000,
    show_default = True,
    help = 'Number of samples per class.'
)
@click.option(
    '--class-shp',
    multiple = True,
    type = ClickPath(exists = True, dir_okay = False, readable = True),
    required = True,
    help = 'Path to shapefile which mark where each class is. Can be given multiple times to make multiclass label. '
        'The order given determines integer indexes (starting from 1) used for representing the classes. '
        'Class index 0 is automatically generated as `other` class. '
        'Note: later shapes will overwrites prior shapes.',
)
@click.option(
    '--preview/--no-preview',
    default = True,
    show_default = True,
    help = 'Whether to output preview of rasterised label.',
)
@click.option(
    '--random-state',
    type = int,
    default = None,
    help = 'Initial RNG state used for sampling.',
)
@click.option(
    '--stratified',
    is_flag = True,
    default = False,
    help = 'Use stratification, i.e., sample each class  proportional to thier occurance in source.',
)
@click.option(
    '-o', '--outdir',
    type = ClickPath(file_okay = False, writable = True),
    default = Path(__file__).parent.joinpath('../{}_dataset'.format(datetime.datetime.today().strftime('%Y%m%dT%H%M%S'))),
    help = 'Directory to store output datasets. [defaults: ../{today}_dataset]',
)
@click.option(
    '--dry-run',
    is_flag = True,
    default = False,
)
@click.option(
    '--train-set-only',
    is_flag = True,
    default = False,
    help = 'Only produce training set.'
)
@click.option(
    'use_dask',
    '--dask/--no-dask',
    default = True,
    show_default = True,
    help = 'Whether to load from datacube lazily with Dask.'
)
@click.option(
    '--out-format',
    type = click.Choice(['npz','nc']),
    multiple = True,
    default = ['npz'],
    show_default = True,
    help = 'Format of output datasets. Can be given multiple times.',
)
@click.option(
    '--format-xy-vars/--format-band-vars',
    default = False,
    show_default = True,
    help = 'Whether to concatenate all bands together as `x` variable; or to keep bands as separate variables.',
)
@click.option(
    '--rgb-preview-step',
    type = int,
    default = None,
    help = 'Positive integer of step size to produce RGB previews. '
        'If not given, no RGB previews will be produced.',
)
@click.option(
    '--select-rgb-bands',
    type = (str,str,str),
    default = ('red','green','blue'),
    help = 'Specify which three bands to use as data for RGB preview.'
)
@click.option(
    '--fill-missing-week',
    is_flag = True,
    default = False,
    help = 'If enabled, missing week data will be filled in with NaN instead of simply being absent from time axis.',
)
@click.option(
    '--rmmeh/--no-rmmeh',
    is_flag = True,
    default = False,
    show_default = False,
    help = 'Apply RMMEH smoothing. '
        '--rmmeh impiles --no-dask',
)
@click.option(
    '--rmmeh-median-window',
    default = 5,
    help = 'The median window size option to pass to rmmeh.',
)
@click.option(
    '--rmmeh-hanning-window',
    default = 5,
    help = 'The hanning window size option to pass to rmmeh.',
)
@click.option(
    '--mask-scl',
    is_flag = True,
    default = False,
    help = 'Mask using SCL. Taking pixels where ((scl > 3) & (scl < 7)) | (scl > 9).',
)
@click.option(
    '--map-label',
    multiple = True,
    type = (int, int),
    help = 'Map class label from 1st label to 2nd label. '
        'Can be given multiple times, they will be operated in the order given. '
        'Mapping to negative number will remove the labelled pixel from dataset.',
)
@click.option(
    '--down-convert/--no-down-convert',
    default = True,
    help = 'Convert output datatype to lower precision before output.',
)
def cli(
    product,
    latitude_extent,
    longitude_extent,
    time_extent,
    band, inverted_band, rmmeh_ignore_band,
    n_sample,
    class_shp,
    preview, rgb_preview_step, select_rgb_bands,
    random_state,
    stratified,
    outdir,
    dry_run,
    train_set_only,
    use_dask,
    out_format, format_xy_vars,
    fill_missing_week,
    rmmeh, rmmeh_median_window, rmmeh_hanning_window,
    mask_scl,
    map_label,
    down_convert,
):
    ## recording command line arguments to file
    
    if outdir is not None:
        outdir.mkdir(parents=True,exist_ok=True)
        with open(outdir.joinpath('cmd'),'a') as f:
            print('===', datetime.datetime.now(), 'dataset generator version', __version__, file = f)
            print(sys.argv, file = f)
            print(' '.join(sys.argv), file = f)
    
    ## argument validation and parsing
    
    if rgb_preview_step is not None and rgb_preview_step <= 0:
        raise ValueError('--rgb-preview-step must be positive integer.')
    
    ## reading image from datacube
    
    _logger.info('obtaining dask object from datacube' if use_dask else 'loading from datacube')
    with datacube.Datacube() as dc:
        ds = dc.load(
            product = product,
            latitude = latitude_extent,
            longitude = longitude_extent,
            time = time_extent,
            measurements = (
                list(set(band) | set(inverted_band) | set(rmmeh_ignore_band))
                if len(band) > 0 else
                None
            ),
            dask_chunks = {
                'time': 1,
                'latitude': 4096,
                'longitude': 4096,
            } if use_dask else None
        )
    _logger.info('obtaining dask object from datacube finished' if use_dask else 'loading from datacube finished')
    
    ## reading shapefiles
    
    ds = rasterise_label(ds, class_shp)
    
    ## mask by SCL
    
    ds = apply_scl(ds, mask_scl)
    
    ## resample to fill missing week
    
    ds = resample_along_time(ds, inverted_band, fill_missing_week)
    
    ## smoothing
    
    if rmmeh:
        if not fill_missing_week:
            raise ValueError('RMMEH can not be applied if --fill-missing-week is not selected.')
        apply_rmmeh(
            ds, inverted_band, rmmeh_ignore_band,
            median_window = rmmeh_median_window,
            hanning_window = rmmeh_hanning_window,
        )
    
    ## down convert after smoothing if precision is no longer needed
    
    if down_convert:
        ds = ds.astype(np.float32)
    
    ## output preview images
    
    if outdir is not None:
        if preview:
            preview_label_rst(ds, outdir)
        if rgb_preview_step is not None:
            if use_dask and rmmeh:
                _logger.info('whole time series will be computed before preview because of RMMEH.')
                for band in select_rgb_bands:
                    _logger.info(f'computing whole {band} time series.')
                    ds.data_vars[band].load()
                    _logger.info(f'computing whole {band} time series finished.')
            preview_timesteps(ds, outdir, select_rgb_bands, rgb_preview_step)
    
    ## reshape in preparation for train_test_split: combine latitude,longitude into single dimension
    
    ds = ds.assign_coords(
        row = ('longitude', range(len(ds.coords['longitude']))),
        col = ('latitude', range(len(ds.coords['latitude']))),
    )
    ds = ds.stack({'location':('latitude','longitude')}).transpose('location', temporal_dim(ds))
    
    ## remapping labels and remove 
    
    # TODO both remapping and sampling should be done before smoothing to avoid unnecessary workload.
    # TODO maybe, separate the dropping of label from mapping of label.
    #     This way, dropping can be done after reshape while mapping can be done before reshape.
    
    for src,dst in map_label:
        if dst < 0:
            ds = ds.drop_isel(location = ds.y == src)
        else:
            ds['y'] = xr.where(ds.y == src, dst, ds.y)

    ## spliting train, test set
    
    datasets = {}
    if train_set_only:
        datasets['train'] = ds
    elif stratified:
        traini, testi = train_test_split(
            np.arange(len(ds.location)),
            stratify = ds.y,
            random_state = random_state,
        )
        datasets['train'] = ds.isel(location = traini)
        datasets['test'] = ds.isel(location = testi)
        del traini, testi
    else:
        equal_per_class_split(ds, n_sample, datasets)
    
    # TODO: How to not duplicate data between `ds` and `datasets`?
    # This isn't a problem if dask was used, but might be if not.
    
    ## Optionally concatenate all data variables together as 'x'
    
    if format_xy_vars:
        for k,subset in datasets.items():
            datasets[k] = convert_to_xy_vars(subset)
    
    ## report
    
    report(datasets)
    
    ## output
    
    if outdir is not None and not dry_run:
        outdir.mkdir(parents = True, exist_ok = True)
        if format_xy_vars:
            if use_dask:
                # by triggering computation via a single dask.compute call,
                # we avoid duplicating common pre-requisite works between datasets
                _logger.info('triggering dask computation pre-writing to files')
                dask_objs = [
                    (k, subset)
                    for k,subset in datasets.items()
                    if dask.is_dask_collection(subset)
                ]
                with dask.config.set(scheduler = 'processes'):
                    computeds = dask.compute(x for _,x in dask_objs)[0]
                for (k,_),computed in zip(dask_objs, computeds):
                    datasets[k] = computed
            _logger.info(f'Writing datasets to {outdir}')
            for k,subset in datasets.items():
                for fm in out_format:
                    globals()['save_dataset_'+fm](k, subset, outdir)
        else:
            save_band_vars_datasets_nc(datasets, ds, outdir)
    
    return locals()


def temporal_dim(ds):
    return 'time' if 'time' in ds.coords else 'week'


def rasterise_label(ds, class_shp):
    _logger.info('raterising class labels')
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
    _logger.info('raterising class labels finished')
    return ds


def apply_scl(ds, mask_scl = True):
    """Use SCL to mask off poor quality pixels as NaN and convert datatype to float32
    
    NOTE: new code should use kea.broker.sentinel2.ard_dataset.from_ingested module
    instead of writing ARD preparation on its own.
    The reason this module is preparing ARD on its own is due to legacy reason:
    it does NOT scale the band values (such as reflectance to be in range [0,1]).
    """
    ds = ds.copy()
    quantitative_bands = list(ds.drop_vars('SCL').data_vars)
    reflectance_bands = list(ds.drop_vars(['SCL', 'WVP', 'AOT'], errors='ignore').data_vars)
    scl = ds.data_vars['SCL']
    
    with xr.set_options(keep_attrs = True):
        ds[quantitative_bands] = ds[quantitative_bands].astype(np.float32).where(lambda x: x != 0)
        
        if mask_scl:
            _logger.info('masking poor quality pixel with SCL')
            mask = ((scl > 3) & (scl < 7)) | (scl > 9)
            ds[quantitative_bands] = ds[quantitative_bands].where(mask)
        
        ds[reflectance_bands] = xr.concat((
            ds[reflectance_bands].sel(time = slice(None, '2022-01-24')),
            ds[reflectance_bands].sel(time = slice('2022-01-25', None)) - 1000,
        ), 'time').where(lambda x: (0 <= x) & (x <= 10000))
    
    _logger.info('masking with SCL finished')
    return ds


def resample_along_time(ds, inverted_band, fill_missing_week = True):
    _logger.info('group by week & fill missing week')
    ds = ds.assign_coords(
        week = ds.time.dt.strftime('%Y-%U'),
    )
    # TODO it does not make logical sense to take max value of SCL, proper way maybe median
    # TODO because of resample, SCL dtype is forced to become float to support np.nan
    if fill_missing_week:
        ds_ = xr.merge((
            ds[list(inverted_band)].resample({'time':pd.offsets.Week(weekday=6)}).min(),
            ds.drop_vars(inverted_band).resample({'time':pd.offsets.Week(weekday=6)}).max(),
        ))
    else:
        ds_ = xr.merge((
            ds[list(inverted_band)].groupby('week').min(),
            ds.drop_vars(inverted_band).groupby('week').max(),
        ))
    _logger.info('group by week & fill missing week finished')
    return ds_


def apply_rmmeh(ds, inverted_band = (), rmmeh_ignore_band = (), median_window = 5, hanning_window = 5):
    """Replace band-variables with its RMMEH-smoothed version."""
    _logger.info('rmmeh smoothing')
    for k,da in ds.data_vars.items():
        if k in rmmeh_ignore_band:
            continue
        ts_chunk = da.chunk({'time': da.sizes['time']})
        ds[k] = ts_chunk.copy(
            deep = False,
            data = ts_chunk.data.map_blocks(
                nd_rmmeh,
                dtype = np.float,
                meta = ts_chunk.data,
                # the following is kwargs to nd_rmmeh
                dim = ts_chunk.get_axis_num('time'),
                median_window = median_window,
                hanning_window = hanning_window,
                use_max = k not in inverted_band,
            ),
        )
    _logger.info('rmmeh smoothing finished')
    return ds


def preview_label_rst(ds, outdir):
    preview_path = outdir.joinpath('preview_of_label_rst.png')
    _logger.info(f'creating preview image of rasterized labels at {preview_path}')
    outdir.mkdir(parents = True, exist_ok = True)
    ds.y.plot.imshow(origin = 'upper')
    plt.tight_layout()
    plt.savefig(preview_path)
    plt.close()
    _logger.info(f'preview image of rasterized labels saved to {preview_path}')


def preview_timesteps(ds, outdir, select_rgb_bands, rgb_preview_step = 1):
    _logger.info('producing RGB previews')
    composite_name = '-'.join(select_rgb_bands)
    fns = []
    sel = xr.concat(
        ds[list(select_rgb_bands)].isel(time = slice(None, None, rgb_preview_step)).data_vars.values(),
        'band',
    ).rename(composite_name)
    for t,im in sel.groupby('time'):
        plt.figure()
        im.plot.imshow(origin='upper',robust=True)
        fn = 'preview_of_{}_{}.png'.format(
            composite_name,
            pd.to_datetime(t).strftime('%Y%m%dT%H%M%S'),
        )
        plt.savefig(outdir.joinpath(fn))
        plt.close()
        fns.append(fn)
    _logger.info('producing RGB previews finished. Forking gif creation process.')
    subprocess.Popen([
        'convert', '-delay', '100', '-loop', '0',
        *fns, '{}_timelapse.gif'.format(composite_name),
    ], cwd = outdir)
    return sel


def equal_per_class_split(ds, n_sample, datasets = None):
    """Split train-test sets such that all class, in any set, has the same number of samples.
    
    ds -- xarray.Dataset
        source dataset
    n_sample -- int
        Limits the maximum number of samples in a class in a set.
        The actual number of samples may be lowered if there is not enough samples.
    datasets -- dict
        If given, update this dictionary with the result.
    
    return
        dict of `xarray.Dataset`s with keys 'train' and 'test'
    """
    if datasets is None:
        datasets = {}
    unique_labels_count = ds.y.groupby('y').count()
    unique_labels = unique_labels_count.y
    # adjust number of sample per class if there is not enough samples
    n_sample = min(n_sample, unique_labels_count.min().item()//2)
    splits = [
        train_test_split(
            np.nonzero(ds.y.values == l)[0],
            test_size = n_sample,
            train_size = n_sample,
        )
        for l in unique_labels.values
    ]
    for i,k in enumerate(('train','test')):
        datasets[k] = ds.isel(
            location = [
                x
                for split in splits
                for x in split[i]
            ]
        )
    return datasets


def report(datasets):
    _logger.info('Dataset extraction has been planned:')
    for k,ds in datasets.items():
        print('---', k, 'set:', file = sys.stderr)
        print(ds, file = sys.stderr)
        print('with following label counts:', file = sys.stderr)
        print(ds.y.groupby(ds.y).count(), file = sys.stderr)


def convert_to_xy_vars(ds):
    if 'x' in ds.data_vars or 'y' in ds.data_vars:
        _logger.info('Already have x or y as data variable already.')
        return ds
    return xr.concat(
        ds.data_vars.values(),
        'band',
    ).rename('x').transpose('location','band',temporal_dim(ds)).to_dataset().reset_coords(['y']).assign_coords(
        band = ('band',list(ds.data_vars.keys()))
    )


def save_dataset_npz(name, ds, outdir):
    path = outdir.joinpath(name + 'Set.npz')
    np.savez(
        path,
        # all data vars
        **{
            k: da.values
            for k,da in ds.data_vars.items()
        },
        # dim order of each vars
        **{
            k+'_dims': da.dims
            for k,da in ds.data_vars.items()
        },
        # all coords at dataset level
        **{
            k: da.values
            for k,da in ds.coords.items()
        },
    )
    _logger.info(f'{path} written')


def save_dataset_nc(name, ds, outdir):
    path = outdir.joinpath(name + 'Set.nc')
    ds.reset_index('location').to_netcdf(path)
    _logger.info(f'{path} written')


def save_band_vars_datasets_nc(datasets, ds, outdir):
    _logger.info(f'Writing datasets to {outdir}')
    _logger.info(f'Writing in append mode, only .nc file is supported.')
    var_order = list(ds.data_vars)
    set_order = list(datasets)
    path_order = [outdir.joinpath(name + 'Set.nc') for name in set_order]
    for v in var_order:
        computeds = dask.compute(datasets[k][[v]] for k in set_order)[0]
        for path,computed in zip(path_order, computeds):
            computed.reset_index('location').to_netcdf(path, 'a' if path.exists() else 'w')
            _logger.info(f'{v} appended to {path}')
