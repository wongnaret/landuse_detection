import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import matplotlib.pyplot as plt


@click.command()
@click.option(
    '--out-template',
    default = 'investigate_na/{ard_file.stem}.png',
    help = 'Template for naming output plots',
)
@click.pass_context
def investigate_na(
    ctx, out_template,
):
    """Visualise where NaN are"""
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    ds = ctx.obj['ds']
    
    # TODO also check inf, -inf
    
    _logger.info(f'loading dataset from {ctx.obj["ard_file"]}')
    ds.load()
    _logger.info(f'dataset {ctx.obj["ard_file"]} loaded')
    
    isnull = ds.isnull().to_array('band', 'isnull').sum('band')
    
    fig, axes = plt.subplots(
        2, 1,
        figsize = (12, 10),
    )
    plt.sca(axes[0])
    isnull.sum('location').plot.line()
    plt.sca(axes[1])
    isnull.sum('time').plot.line()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    _logger.info(f'{outpath} created')
    
    return locals()


@click.command()
@click.option(
    '--out-template',
    default = '{ard_file.stem}_na_interpolated.nc',
    help = 'Template for naming output NaN interpolated .nc file',
)
@click.pass_context
def interpolate_na(
    ctx, out_template,
):
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    ds = ctx.obj['ds']
    
    ds = ds.interpolate_na('time', method = 'linear', use_coordinate = False).bfill('time').ffill('time')
    # at this point, only locations with full NaN at all times is still leftover
    ds = ds.dropna('location', 'any')
    
    _logger.info(f'writing to {outpath}')
    ds.to_netcdf(outpath)
    _logger.info(f'{outpath} created')
    
    return locals()
