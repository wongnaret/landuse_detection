import logging
_logger = logging.getLogger(__name__)
import pickle
# 3rd party modules
import click
import dask_ml.cluster


@click.command()
@click.option(
    '--out-template',
    default = '{encoded_file.stem}_spectral_k{ncluster}{tag[ncomponent]}{tag[limit]}.pkl',
    help = 'Template for naming output pickle file containing the resulting spectral clustering model',
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
@click.pass_context
def calculate_spectral(
    ctx, out_template,
    limit, ncomponent, ncluster,
):
    """Performs spectral clustering"""
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
    
    _logger.info('fitting spectral clustering model')
    
    spectral_model = dask_ml.cluster.SpectralClustering(
        
        # these are defaults ... which shouldn't work
        #n_clusters = 2,
        #n_components = 100,
        
        ncluster,
        persist_embedding = True,
        n_jobs = 5,
        kmeans_params = dict(n_clusters = ncluster),
        affinity = 'rbf',
    )
    labels = spectral_model.fit_predict(encoded.pca_encoded_ard)
    
    _logger.info('fitting spectral clustering model finished')
    
    # TODO just save the labels
    
    return locals()
