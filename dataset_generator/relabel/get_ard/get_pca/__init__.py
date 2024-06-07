import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
# custom modules
from .calculate import calculate_pca
from .load import load_pca


@click.group(invoke_without_command = True)
@click.pass_context
def get_pca(
    ctx,
):
    ds = ctx.obj['ds']
    
    reshaped = (
        ds.to_array('band', 'sample_feature')
        .stack(feature = ('band', 'time'))
        .chunk(dict(feature = -1))
        .transpose('location', 'feature')
    )
    
    ctx.obj['reshaped'] = reshaped
    return locals()


get_pca.add_command(calculate_pca)
get_pca.add_command(load_pca)
