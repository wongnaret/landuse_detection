import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import xarray as xr
# custom modules
from kea.clickpath import ClickPath
from .calculate_kmean import calculate_kmean
from .load_kmean import load_kmean
#from .calculate_spectral import calculate_spectral
from .plot_scatter import plot_scatter


@click.group(invoke_without_command = True)
@click.option(
    '--encoded-file', '--encoded', '--file',
    type = ClickPath(exists = True, dir_okay = False),
    required = True,
    help = 'Input netCDF (.nc) file containing encoded ARD raster.',
)
@click.pass_context
def get_encoded(
    ctx, encoded_file,
):
    _logger.info(f'opening {encoded_file}')
    encoded = xr.open_dataset(encoded_file, chunks = {'component': -1, 'location': 'auto'})
    
    ctx.obj['encoded_file'] = encoded_file
    ctx.obj['encoded'] = encoded
    return locals()


get_encoded.add_command(calculate_kmean)
get_encoded.add_command(load_kmean)
#get_encoded.add_command(calculate_spectral)
get_encoded.add_command(plot_scatter)
