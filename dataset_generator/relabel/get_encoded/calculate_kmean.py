import logging
_logger = logging.getLogger(__name__)
import pickle
# 3rd party modules
import click
import dask_ml.cluster
import sklearn.cluster


@click.command()
@click.option(
    '--out-template',
    default = '{encoded_file.stem}_kmean_k{ncluster}{tag[ncomponent]}{tag[limit]}.pkl',
    help = 'Template for naming output pickle file containing the resulting K-mean model',
)
@click.option(
    '--limit',
    type = int,
    help = 'Limit number of location used',
)
@click.option(
    '--ncomponent',
    type = int,
    help = 'Limit number of PCA component used [default: use all components available]',
)
@click.option(
    '--ncluster',
    type = int,
    default = 20,
    help = 'Number of cluster',
)
@click.option(
    'use_dask', '--dask/--no-dask',
    is_flag = True,
    default = True,
    show_default = True,
    help = 'Whether to use `dask_ml.cluster` or `sklearn.cluster`',
)
@click.pass_context
def calculate_kmean(
    ctx, out_template, use_dask,
    limit, ncomponent, ncluster,
):
    ctx.obj.setdefault('tag', {}).update(
        limit = '' if limit is None else f'_limit[{limit}]',
        ncomponent = '' if ncomponent is None else f'_ncomp[{ncomponent}]',
    )
    ctx.obj['ncluster'] = ncluster
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    encoded = ctx.obj['encoded']
    
    if ncomponent is not None:
        encoded = encoded.isel(component = slice(None, ncomponent))
    if limit is not None:
        encoded = encoded.isel(location = slice(None, limit))
    
    _logger.info('fitting K-means model')
    
    if use_dask:
        kmean_model = dask_ml.cluster.KMeans(ncluster)
        kmean_model.fit(encoded.pca_encoded_ard)
    else:
        kmean_model = sklearn.cluster.KMeans(ncluster)
        kmean_model.fit(encoded.pca_encoded_ard.values)
    
    _logger.info('fitting K-means model finished')
    
    # TODO: because kmean_model.labels_ is a dask array,
    # it causes original input array to also get written out when pickled.
    with open(outpath, 'wb') as f:
        pickle.dump(kmean_model, f)
    _logger.info(f'K-means model written to {outpath}')
    
    return locals()
