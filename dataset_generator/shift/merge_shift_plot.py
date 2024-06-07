"""Find location intersection of multiple npz datasets, sample sugarcane, shift and plot."""
from pathlib import Path
import click
import numpy as np
import xarray as xr
import seaborn as sns
# custom modules
import npz_to_xr
import shapeshift_npz
from cmd_log import cmd_log


# https://github.com/pallets/click/issues/405#issuecomment-470812067
class ClickPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


@click.group(
    invoke_without_command = True,
    help = 'Find location intersection of multiple npz datasets, sample sugarcane, shift and plot.'
)
@click.option(
    '--npz',
    required = True,
    multiple = True,
    type = (str, ClickPath(exists = True, dir_okay = False)),
    help = 'Add dataset to the analysis. '
        'First component is a display name for the dataset. '
        'Second component is path to .npz file.',
)
@click.option(
    '-o',
    '--outdir',
    type = ClickPath(file_okay = False),
    default = Path('output'),
    help = 'The directory for storing outputs.',
)
@click.option(
    '-b',
    '--search-begin',
    show_default = True,
    help = 'The earliest crop cycle can start.',
)
@click.option(
    '-e',
    '--search-end',
    show_default = True,
    help = 'The latest crop cycle can start.',
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
def cli(
    npz, outdir,
    search_begin, search_end,
    plot_y_begin, plot_y_end,
):
    sns.set()
    cmd_log(outdir)
    dsns = [ name for name,_ in npz ]
    dss = [ npz_to_xr.read(f) for _,f in npz ]
    dss = xr.align(*dss)
    for ds in dss:
        shapeshift_npz.append_crop_age(ds, search_begin, search_end)
    
    # use same random state for each sampling & plot call
    random_state = np.random.get_state()
    for name, ds in zip(dsns, dss):
        np.random.set_state(random_state)
        shapeshift_npz.plot_compare_shifted_sugarcane(ds, outdir = outdir, ylim = (plot_y_begin, plot_y_end), source = name)
    
    return locals()


if __name__ == '__main__':
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret,dict):
        globals().update(cli_ret)
