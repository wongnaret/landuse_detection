from pathlib import Path
import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
import click
from dask.distributed import Client
# custom modules
from kea.clickpath import ClickPath
from .get_ard import get_ard
from .get_encoded import get_encoded


@click.group(invoke_without_command = True)
@click.option(
    '--out-base', '--out',
    type = ClickPath(file_okay = False),
    default = Path(),
    help = 'Base directory to output files to. [default: current directory]',
)
@click.option(
    '--dask-port', '--port',
    type = int,
    help = 'Start distributed dask scheduler on the given port',
)
@click.option(
    '--dry-run',
    is_flag = True,
    help = 'Avoid costly processing as much as possible while just logging what would have been done. '
        'Exact behaviour depends on subcommands.',
)
@click.pass_context
def cli(
    ctx, out_base, dask_port, dry_run,
):
    """Toolbox for ground truth relabelling according to structure obtained from unsupervised learning.
    
    This is the top-most command group which provides generic configuration options."""
    ctx.ensure_object(dict)
    
    if dask_port is not None:
        client = Client(
            dashboard_address = f':{dask_port}',
            n_workers = 12,
            threads_per_worker = 1,
        )
        _logger.info(f'dask dashboard available at {client.dashboard_link}')
        ctx.obj['client'] = client
    
    ctx.obj['out_base'] = out_base
    ctx.obj['dry_run'] = dry_run
    return locals()


cli.add_command(get_ard)
cli.add_command(get_encoded)
