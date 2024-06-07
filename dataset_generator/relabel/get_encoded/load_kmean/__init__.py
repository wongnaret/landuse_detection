import logging
_logger = logging.getLogger(__name__)
import pickle
# 3rd party modules
import click
import xarray as xr
import dask
import dask.array
# custom modules
from kea.clickpath import ClickPath
from .similarity import compare_similarity


@click.group(invoke_without_command = True)
@click.option(
    '--kmean-file', '--kmean', '--file',
    type = ClickPath(exists = True, dir_okay = False),
    required = True,
    help = 'Input pickle (.pkl) file containing K-mean clustering model of encoded ARD. ',
)
@click.pass_context
def load_kmean(
    ctx, kmean_file,
):
    _logger.info(f'opening {kmean_file}')
    
    with open(kmean_file, "rb") as f:
        kmean_model = pickle.load(f)
    kmean_ds = kmean_model_to_dataset(kmean_model)
    
    ctx.obj['kmean_file'] = kmean_file
    ctx.obj['kmean_model'] = kmean_model
    ctx.obj['kmean_ds'] = kmean_ds
    return locals()


def kmean_model_to_dataset(kmean_model):
    cluster_label = kmean_model.labels_
    if not dask.is_dask_collection(cluster_label):
        cluster_label = dask.array.from_array(cluster_label)
        
    kmean_ds = xr.Dataset(
        data_vars = dict(
            cluster_center = (('cluster', 'component'), kmean_model.cluster_centers_),
            cluster_label = (('location',), cluster_label),
        ),
    )
    kmean_ds = kmean_ds.assign_coords(
        cluster = kmean_ds.cluster,
        # assumes component is always picked continuously from the most significant ones
        component = kmean_ds.component,
    )
    return kmean_ds


load_kmean.add_command(compare_similarity)
