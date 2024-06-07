from pathlib import Path
import click
import xarray as xr
import numpy as np
import pandas as pd


# https://github.com/pallets/click/issues/405#issuecomment-470812067
class ClickPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


def pyi(
    data_dir,
    begin_week, end_week,
):
    datasets = load_data(data_dir)
    crop_start, crop_starti, mask, masked_ndvi = find_crop_start(
        datasets['test'], begin_week, end_week,
    )
    datasets['test_shifted'] = produce_shifted(datasets['test'], crop_starti, begin_week)
    return locals()


def produce_shifted(ds, crop_starti, begin_week):
    shift = crop_starti + ds.indexes['weeks'].get_loc(begin_week)
    
    # interesting stats for debugging:
    #maxshift, minshift = shift.max().item(), shift.min().item()
    #oldlength = ds.sizes['weeks']
    #newlength = (maxshift - minshift) * 2 + oldlength
    
    base = ds.sel(location = shift.indexes['location'])
    base = base.assign_coords(
        weeks = ('weeks', range(len(base.coords['weeks']))),
    )
    # iterate through each series, shift, then recombine
    def shifted_ts():
        for li in range(base.sizes['location']):
            ts = base.isel(location = li)
            ts.coords['weeks'] = ts.coords['weeks'] - shift.isel(location = li).item()
            yield ts
    return xr.concat(shifted_ts(),'location')
    

cli = pyi
@click.group(
    invoke_without_command = True,
)
@click.option(
    '--data-dir',
    type = ClickPath(exists = True, file_okay = False),
    default = '../generator/data-for-shift',
    show_default = True,
)
@click.option(
    '--begin-week',
    default = '2017-45',
    show_default = True,
    help = '{%Y year}-{%U week-of-year} where search for crop start will begin.'
)
@click.option(
    '--end-week',
    default = '2018-24',
    show_default = True,
    help = '{%Y year}-{%U week-of-year} where search for crop start will end.'
)
def cli(*args,**kwargs):
    return pyi(*args,**kwargs)


def load_data(data_dir):
    data_dir = Path(data_dir)
    datasets = {
        nc.stem.rsplit('Set',1)[0]: xr.open_dataset(nc).set_index(
            append = True,
            location = ['latitude','longitude'],
        )
        for nc in data_dir.glob('*Set.nc')
    }
    for ds in datasets.values():
        ds.load()
    return datasets


def find_crop_start(ds,begin_week, end_week):
    da = ds['x'].sel(weeks=slice(begin_week, end_week))
    red = da.sel(band = 'red') / 1e4
    nir = da.sel(band = 'nir') / 1e4
    ndvi = (nir-red) / (nir+red)
    # filter out by SCL
    scl = da.sel(band = 'SCL')
    mask = ((scl > 3) & (scl < 7)) | (scl > 9)
    masked_ndvi = ndvi.where(mask).dropna('location',how='all')
    crop_starti = masked_ndvi.argmin('weeks', skipna = True)
    crop_start = da['weeks'][crop_starti].drop(('band','weeks'))
    
    print('Distribution of crop_start')
    print(crop_start.to_pandas().value_counts().sort_index())
    
    return crop_start, crop_starti, mask, masked_ndvi
    """# explore distribution of min NDVI VS distribution of SCL
    from pprint import pprint
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    pprint(list(zip(
        *np.unique(crop_start.values, return_counts = True)
    )))
    pprint(list(zip(
        *np.unique(da.sel(band = 'SCL', weeks = '2019-00').values, return_counts = True)
    )))
    #ndvi.assign_coords(
    #    wi = ('weeks', range(len(ndvi.coords['weeks']))),
    #).set_index({'weeks':'wi','location':'location'}).plot.line(x='weeks')
    tmp = ndvi.to_dataframe().reset_index().pivot(
        index='weeks',
        columns=('latitude','longitude'),
        values='x',
    )
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=tmp.iloc[:,:100])
    plt.savefig('ndvi_at_crop_start.png')
    plt.close()
    ## scl table around suspected crop start
    scl_table = da.sel(band = 'SCL').to_pandas().apply(pd.Series.value_counts)
    scl_table.to_csv('scl_table.csv')
    """


# Example:
# plot_ndvi( datasets['test_shifted'].isel(location=slice(50)) )
def plot_ndvi(ds):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    nir = ds['x'].sel(band='nir') / 10000
    red = ds['x'].sel(band='red') / 10000
    ndvi = (nir-red) / (nir+red)
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=ndvi.values)
    plt.savefig('ndvi.png')
    plt.close()


def ndvi_at_crop_start_hist(masked_ndvi,bins=20):
    df = pd.DataFrame(np.histogram(masked_ndvi.min('weeks', skipna = True).values, bins)).T
    return df

if __name__ == '__main__':
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret,dict):
        globals().update(cli_ret)