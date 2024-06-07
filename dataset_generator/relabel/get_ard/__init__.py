from pathlib import Path
import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import xarray as xr
# custom modules
from kea.clickpath import ClickPath
from .ard_prep import investigate_na, interpolate_na
from .get_pca import get_pca


def get_default_ard_file():
    try:
        return next(Path().glob('*.nc'))
    except Exception:
        pass
    return


@click.group(invoke_without_command = True)
@click.option(
    '--ard-file', '--ard', '--file',
    type = ClickPath(exists = True, dir_okay = False),
    default = get_default_ard_file(),
    required = True,
    help = 'Input netCDF (.nc) file containing ARD raster. [default: the first .nc found in current directory]',
)
@click.pass_context
def get_ard(
    ctx, ard_file,
):
    """Loads Analysis Ready Data (ARD) to use in further subcommands."""
    _logger.info(f'opening {ard_file}')
    ds = xr.open_dataset(ard_file, chunks = {...: 'auto', 'location': 3000})
    # discard SCL as it is not quantitative, thus not compatible with PCA
    # discard AOT, WVP as they contains bugs (cloud mask was performed without smoothing to refill the NaNs)
    ds = ds.drop_vars(['SCL','WVP','AOT'], errors = 'ignore')
    
    ctx.obj['ard_file'] = ard_file
    ctx.obj['ds'] = ds
    return locals()


get_ard.add_command(investigate_na)
get_ard.add_command(interpolate_na)
get_ard.add_command(get_pca)
