import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
import xarray as xr


@click.command(name = 'encode')
@click.option(
    '--out-template',
    default = '{ard_file.stem}_encoded{tag[whiten]}.nc',
    help = 'Template for naming output netCDF file containing ARD re-encoded by PCA',
)
@click.option(
    '--whiten/--no-whiten',
    default = False,
    show_default = True,
    help = 'Whether to normalise encoded data so each component has similar range and spread. '
        'Information about relative variance between components will be lost.',
)
@click.pass_context
def encode_ard_with_pca(
    ctx, out_template, whiten,
):
    ctx.obj.setdefault('tag', {}).update(
        whiten = '_whiten' if whiten else '',
    )
    outpath = ctx.obj['out_base'] / out_template.format_map(ctx.obj)
    outpath.parent.mkdir(parents = True, exist_ok = True)
    model = ctx.obj['model']
    pca_ds = ctx.obj['pca_ds']
    reshaped = ctx.obj['reshaped']
    
    model.whiten = whiten
    
    _logger.info('planning to encode ard with pca')
    encoded = model.transform(reshaped)
    encoded = xr.DataArray(
        encoded,
        dims = ('location', 'component'),
        coords = copy_coords(('location', 'component'), (reshaped, pca_ds)),
        name = 'pca_encoded_ard',
        attrs = dict(
            source_ard_file = str(ctx.obj['ard_file'].resolve()),
            pca_title = ctx.obj['pca_title'],
        ),
    ).to_dataset()
    
    if ctx.obj['dry_run']:
        _logger.info(f'--dry-run skips writing to {outpath}')
    else:
        _logger.info(f'writing to {outpath}')
        encoded.to_netcdf(outpath)
        _logger.info(f'{outpath} written')
    
    ctx.obj['encoded'] = encoded
    return locals()


def copy_coords(dims, sources):
    """Make dictionary of all coordinates from `sources` which lies inside a set of `dims`."""
    coords = {}
    for src in sources:
        for coord_name, coord in src.coords.items():
            if all(dim in dims for dim in coord.dims):
                coords[coord_name] = coord
    return coords
