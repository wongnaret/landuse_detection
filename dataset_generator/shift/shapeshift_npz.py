import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import click
import matplotlib.pyplot as plt
import matplotlib.ticker
# custom modules
import npz_to_xr
import read_nc
from cmd_log import cmd_log


# https://github.com/pallets/click/issues/405#issuecomment-470812067
class ClickPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


@click.group(
    invoke_without_command = True,
    help = 'Dectect crop start and shift time axis to align crop age.'
)
@click.option(
    '--npz',
    type = ClickPath(exists = True, dir_okay = False),
    help = 'The source dataset in npz format.',
    # example: /home/apiwat/git/lstm/lstm_dataset/generator/kc_full_data/trainSet.npz
)
@click.option(
    '-o',
    '--outdir',
    type = ClickPath(file_okay = False),
    default = Path('output'),
    help = 'The directory for storing outputs.',
)
@click.option(
    '--source-label',
    default = '[unknown-source]',
    help = 'Display name for the source of dataset, e.g., farm name.',
)
@click.option(
    '-b',
    '--search-begin',
    #default = '2019-11',
    show_default = True,
    help = 'The earliest crop cycle can start.',
)
@click.option(
    '-e',
    '--search-end',
    #default = '2020-4',
    show_default = True,
    help = 'The latest crop cycle can start.',
)
@click.option(
    '--plot/--no-plot',
    default = True,
    help = 'Whether to output plot of shifted data or not.',
)
@click.option(
    '--plot-y-begin',
    type = float,
    help = 'The begining of y-axis for plotting',
)
@click.option(
    '--plot-y-end',
    type = float,
    help = 'The ending of y-axis for plotting',
)
@click.option(
    '--do-shift',
    is_flag = True,
    default = False,
    help = 'Actually make `shifted` dataset with crop_age as dimension replacing time. '
        'Without this flag, the crop_age is calculated but not actually used as dimension.',
)
def cli(
    npz, outdir, source_label,
    search_begin, search_end,
    plot, plot_y_begin, plot_y_end,
    do_shift,
):
    if outdir is not None:
        cmd_log(outdir)
    ds = load_shifted(npz, search_begin, search_end, do_shift)
    if do_shift:
        ds, shifted = ds
    if plot:
        plot_compare_shifted_sugarcane(ds, outdir = outdir, ylim = (plot_y_begin, plot_y_end), source = source_label)
    return locals()


def append_crop_age(ds, search_begin, search_end):
    """modify dataset in-place adding ndvi, crop_start, crop_age data variables."""
    ds['ndvi'] = ndvi(ds)
    ds['crop_start'], ds['crop_start_i'] = crop_start(ds, time_slice = slice(search_begin, search_end))
    ds['crop_start'].attrs['search_extent'] = (search_begin, search_end)
    ds['crop_age'] = ds.time - ds.crop_start


def ndvi(ds):
    red = ds.x.sel(band = 'red') / 1e4
    nir = ds.x.sel(band = 'nir') / 1e4
    return (nir-red) / (nir+red)


def crop_start(ds, time_slice = slice('2019-11','2020-4'), return_index = True):
    # TODO determine search range automatically
    # ndvi in growing period
    ndvi_igp = ds.ndvi.sel(time = time_slice)
    min_index = ndvi_igp.argmin('time')
    if return_index:
        earliest_index = ds.indexes['time'].get_loc( ndvi_igp.indexes['time'][0] )
        return ndvi_igp.coords['time'][min_index], min_index + earliest_index
    else:
        return ndvi_igp.coords['time'][min_index]


def load_shifted(filename, search_begin, search_end, do_shift = True):
    """
    filename: str or path-like
        Can be either .npz or .nc
        File extension will determine the loading method.
    """
    filename = Path(filename)
    if filename.suffix == 'npz':
        ds = npz_to_xr.read(filename)
    else:
        # let xarray handle both .nc and the else case
        # since it is possible xr.open_dataset will be cleverer
        ds = read_nc.read(filename, xy_vars = True)
    append_crop_age(ds, search_begin, search_end)
    if do_shift:
        return ds, concat_shift(ds)
    return ds


def merge_shift(ds, varname = 'x'):
    import warnings
    warnings.warn(
        'This method is painfully slow and consumes a lot of memeory. '
        'It has been left here for small experimentation only. '
        'Use concat_shift instead.'
    )
    return xr.merge((
        v.assign_coords(crop_age = v.crop_age.squeeze(drop = True)).set_index(time = 'crop_age')
        for _,v in ds.data_vars[varname].assign_coords(crop_age = ds.crop_age) \
            .drop_vars(('time','weeks')).groupby('location')
    )).rename(time='crop_age')


def concat_shift(ds, varname = 'x'):
    mini = ds.crop_start_i.min().item()
    maxi = ds.crop_start_i.max().item()
    shifted = xr.concat(
        (
            v.pad(time = (lpad,rpad))
            for (_,v),lpad,rpad in zip(
                ds['x'].drop_vars(('time','weeks'), errors = 'ignore').groupby('location'),
                maxi - ds.crop_start_i,
                ds.crop_start_i - mini,
            )
        ),
        'location',
        coords='minimal', compat='override', join='override', combine_attrs='override',
    ).rename(time='crop_age')
    # set coordinate for crop_age dimension
    shifted['crop_age'] = pd.timedelta_range(
        (ds.time.min() - ds.crop_start.max()).item(),
        (ds.time.max() - ds.crop_start.min()).item(),
        freq = '7D', # Can't transfer frequency description from ds.time, since W-SUN 'is not a fixed frequency'
    )
    return shifted.to_dataset().assign(y = ds.y, crop_start = ds.crop_start)


def plot_compare_shifted_sugarcane(ds, outdir = None, nsample = 100, ylim = (-0.1,1.), source = '[unknown-source]'):
    sugarcane_sample = ds.sel(location = (ds.y==1))
    sugarcane_sample = sugarcane_sample.isel(
        location = np.random.choice(
            len(sugarcane_sample.location),
            size = nsample,
            replace = False,
        )
    )
    da = sugarcane_sample.ndvi.assign_coords(
        crop_age_week = sugarcane_sample.crop_age.dt.days // 7
    )
    fig, axes = plt.subplots(nrows = 2, figsize = (14,6))
    
    # original plot
    plt.sca(axes[0])
    for i in range(len(sugarcane_sample.location)):
        da.isel(location = i).plot.line(x = 'time', alpha = 0.5)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_title('Original'.format(nsample))
    axes[0].xaxis.set_minor_locator(matplotlib.dates.MonthLocator()) # tick every month
    plt.xticks(rotation = 0, ha = 'center')
    axes[0].xaxis.grid(True, which='both')
    
    # shifted plot
    plt.sca(axes[1])
    for i in range(len(sugarcane_sample.location)):
        da.isel(location = i).plot.line(x = 'crop_age_week', alpha = 0.5)
    if ylim is not None:
        axes[1].set_ylim(*ylim)
    axes[1].set_title(
        'Aligned by [predicted] crop age' +
        (
            '\nsearching between {} and {}'.format(*sugarcane_sample['crop_start'].attrs['search_extent'])
            if 'search_extent' in sugarcane_sample['crop_start'].attrs else ''
        )
    )
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(4)) # tick every 4 weeks
    axes[1].xaxis.grid(True, which='both')
    
    # figure wide setting
    fig.suptitle('{} random samples of sugarcane NDVI time series from {}'.format(nsample, source), fontsize = 16)
    fig.tight_layout()
    
    # output
    if outdir is not None:
        outdir.mkdir(parents = True, exist_ok = True)
        plt.savefig(outdir.joinpath(source + '_shifted_ndvi.png'))
        plt.close()
    return fig, axes, sugarcane_sample


if __name__ == '__main__':
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret,dict):
        globals().update(cli_ret)
