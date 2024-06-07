import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import dask.array
import xarray as xr
import matplotlib.pyplot as plt


@click.group(invoke_without_command = True)
@click.option(
    '--truth-nlabel',
    type = int,
    default = 3,
    help = 'Number of ground truth labels',
)
@click.option(
    '--out-template',
    default = '{kmean_file.stem}_truth_vs_cluster_label.png',
    help = 'Template for naming output visualisation of similarity between truth and cluster labels.',
)
@click.option(
    '--preserve-cluster-id',
    is_flag = True,
    help = 'Show original cluster ID after sorting.',
)
@click.pass_context
def compare_similarity(
    ctx, truth_nlabel, out_template,
    preserve_cluster_id,
):
    """Compare similarity between unsupervised K-mean clustering labels and ground truth labels."""
    encoded = ctx.obj['encoded']
    kmean_ds = ctx.obj['kmean_ds']
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    
    hist, _ = dask.array.histogramdd(
        [
            # TODO align chunk better
            encoded.y.data.rechunk(kmean_ds.cluster_label.data.chunks),
            kmean_ds.cluster_label.data,
        ],
        bins = [
            list(range(truth_nlabel + 1)),
            list(range(kmean_ds.sizes['cluster'] + 1)),
        ],
    )
    
    hist = xr.DataArray(
        hist,
        dims = ('truth_label', 'cluster'),
        coords = dict(
            truth_label = list(range(truth_nlabel)),
            cluster = list(map(str, range(kmean_ds.sizes['cluster']))),
        ),
        name = 'truth_vs_cluster_label',
    ).load()
    hist = hist.sortby([
        hist.argmax('truth_label'),
        hist.sum('truth_label'),
    ])
    if not preserve_cluster_id:
        hist = hist.reset_index('cluster')
    
    similarity = (hist.max('truth_label').sum() / hist.sum()).item()
    
    fig, axes = plt.subplots(
        2, 1,
        figsize = (16, 7),
        sharex = True,
    )
    
    plt.sca(axes[0])
    hist.plot.imshow(robust = True, x = 'cluster', y = 'truth_label')
    plt.title('raw frequency')
    
    plt.sca(axes[1])
    (hist / hist.sum('truth_label')).plot.imshow(robust = True, x = 'cluster', y = 'truth_label')
    plt.title('normalised per cluster')
    
    plt.suptitle(f'similarity score between cluster and ground truth labelling: {similarity}')
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    return locals()
