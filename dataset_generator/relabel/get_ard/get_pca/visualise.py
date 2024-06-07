import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@click.command(name = 'visualise')
@click.option(
    '--out-template',
    default = '{pca_title}_{which_plot}.png',
    help = 'Template for naming output image file containing visualisation of PCA',
)
@click.option(
    '--variance-ratio-cutoff', '--cutoff',
    type = float,
    default = .95,
    show_default = True,
    help = 'Only display components with explained variance ratio totalling upto this maximum.'
        'Set to >= 1 to display all components.',
    callback = (lambda ctx, params, value: None if np.isnan(value) else value),
)
@click.pass_context
def visualise_pca(
    ctx, out_template, variance_ratio_cutoff,
):
    sns.set()
    model = ctx.obj['model']
    pca_ds = ctx.obj['pca_ds']
    
    # select most significant components for plotting
    component_slice = (
        None if variance_ratio_cutoff is None else
        slice(None, pca_ds.explained_variance_ratio.cumsum().searchsorted(variance_ratio_cutoff) + 1)
    )
    sig = pca_ds.isel(component = component_slice)
    
    ctx.obj['out_template'] = out_template
    plot_explained_variance_ratio(pca_ds, ctx)
    plot_components(sig, ctx)
    del ctx.obj['out_template']
    
    return locals()


def plot_explained_variance_ratio(ds, ctx):
    outpath = prep_outpath(ctx, 'explained_variance')
    fig, ax = plt.subplots(
        figsize = (12, 3),
    )
    
    ds.explained_variance_ratio.plot.line()
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return


def plot_components(ds, ctx):
    outpath = prep_outpath(ctx, 'components')
    fig, axes = plt.subplots(
        ds.sizes['component'], 1,
        figsize = (12, 3 * ds.sizes['component']),
        sharex = True,
    )
    
    for ax, component_id in zip(axes, ds.component.values):
        plt.sca(ax)
        ds.components.isel(component = component_id).plot.line(x = 'time', hue = 'band')
        ax.get_legend().set_visible(False)
    
    handles = ax.get_legend().legendHandles
    labels = [t.get_text() for t in ax.get_legend().texts]
    fig.legend(handles, labels, loc = 'upper right')
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return


def prep_outpath(ctx, which_plot):
    outpath = ctx.obj['out_base'] / ctx.obj['out_template'].format(
        **ctx.obj,
        which_plot = which_plot,
    )
    outpath.parent.mkdir(parents = True, exist_ok = True)
    return outpath
