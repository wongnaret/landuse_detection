import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@click.command()
@click.option(
    '--out-template',
    default = '{encoded_file.stem}_scatter{tag[component]}{tag[limit]}.png',
    help = 'Template for naming output scatter plot in PCA basis.',
)
@click.option(
    '--limit',
    type = int,
    help = 'Limits number of locations plotted.'
)
@click.option(
    '--ncomponent',
    type = int,
    help = 'Limits number of PCA components plotted. Ignored when --comp is given.',
)
@click.option(
    'components', '--comp',
    type = int,
    multiple = True,
    help = 'Selects only the given components to plot.'
)
@click.pass_context
def plot_scatter(
    ctx, out_template,
    limit, ncomponent, components,
):
    sns.set()
    ctx.obj.setdefault('tag', {}).update(
        limit = '' if limit is None else f'_limit[{limit}]',
        component = (
            f'_comp[{",".join(map(str, components))}]' if components else
            (
                '' if ncomponent is None else
                f'_ncomp[{ncomponent}]'
            )
        ),
    )
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    encoded = ctx.obj['encoded']
    
    if limit is not None:
        encoded = encoded.isel(location = slice(None, limit))
    if components:
        encoded = encoded.isel(component = list(components))
    elif ncomponent is not None:
        encoded = encoded.isel(component = slice(None, ncomponent))
    
    _logger.info('plotting')
    df = encoded.load().pca_encoded_ard.to_pandas()
    df['truth_label'] = pd.Categorical.from_codes(
        encoded.y,
        dtype = pd.CategoricalDtype(categories = ['other', 'rice', 'sugar']),
    )
    sns.pairplot(
        df, hue = 'truth_label',
        diag_kind = 'hist',
        plot_kws = dict(alpha = 0.5),
    )
    plt.savefig(outpath)
    plt.close()
    _logger.info(f'written to {outpath}')
    
    # TODO if seaborn is not scallable enough, ...
    
    return locals()
