import logging
_logger = logging.getLogger(__name__)
import pickle
# 3rd party modules
import click
# custom modules
from kea.clickpath import ClickPath
from .visualise import visualise_pca
from .interpret_pca import pca_model_to_dataset
from .encode import encode_ard_with_pca


@click.group(
    name = 'load',
    invoke_without_command = True,
)
@click.option(
    '--pca-file', '--pca', '--file',
    type = ClickPath(exists = True, dir_okay = False),
    required = True,
    help = 'Input pickle (.pkl) file containing fitted PCA model.',
)
@click.pass_context
def load_pca(
    ctx, pca_file,
):
    ctx.obj['pca_title'] = pca_file.stem
    
    _logger.info(f'loading {pca_file}')
    model = load_pca_from_file(pca_file)
    ctx.obj['model'] = model
    
    reshaped = ctx.obj['reshaped']
    pca_ds = pca_model_to_dataset(model, reshaped)
    ctx.obj['pca_ds'] = pca_ds
    
    return locals()


load_pca.add_command(visualise_pca)
load_pca.add_command(encode_ard_with_pca)


def load_pca_from_file(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    # TODO assert isinstance(model, ...)
    return model
