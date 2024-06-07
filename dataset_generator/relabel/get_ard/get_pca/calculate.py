import logging
_logger = logging.getLogger(__name__)
import pickle
# 3rd party modules
import click
import dask_ml.decomposition
# custom modules
from .visualise import visualise_pca
from .interpret_pca import pca_model_to_dataset
from .encode import encode_ard_with_pca


@click.group(
    name = 'calculate',
    invoke_without_command = True,
)
@click.option(
    '--out-template',
    default = '{ard_file.stem}_pca_model{tag[incremental]}{tag[ncomponent]}{tag[limit]}.pkl',
    help = 'Template for naming output pickle file containing the resulting PCA model',
)
@click.option(
    '--limit',
    type = int,
    help = 'Limit number of location used',
)
@click.option(
    '--ncomponent',
    type = int,
    help = 'Number of components to keep',
)
@click.option(
    '--incremental',
    is_flag = True,
    help = 'Whether to use incremental PCA algorithm which use constant memory footprint (independent of progress)',
)
@click.pass_context
def calculate_pca(
    ctx, out_template, limit, ncomponent, incremental
):
    ctx.obj.setdefault('tag', {}).update(
        limit = '' if limit is None else f'_limit[{limit}]',
        ncomponent = '' if ncomponent is None else f'_ncomp[{ncomponent}]',
        incremental = '_inc' if incremental else ''
    )
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    ctx.obj['pca_title'] = outpath.stem
    ds = ctx.obj['ds']
    reshaped = ctx.obj['reshaped']
    
    if limit is not None:
        reshaped = reshaped.isel(location = slice(None, limit))
    
    _logger.info('starting PCA')
    
    model = (
        dask_ml.decomposition.IncrementalPCA
        if incremental else
        dask_ml.decomposition.PCA
    )
    model = model(n_components = ncomponent, whiten = True).fit(reshaped.data)
    
    _logger.info('PCA finished')
    with open(outpath, 'wb') as f:
        pickle.dump(model, f)
    _logger.info(f'PCA model written to {outpath}')
    
    ctx.obj['model'] = model
    pca_ds = pca_model_to_dataset(model, reshaped)
    ctx.obj['pca_ds'] = pca_ds
    
    return locals()


calculate_pca.add_command(visualise_pca)
calculate_pca.add_command(encode_ard_with_pca)
