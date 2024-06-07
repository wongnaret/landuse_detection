import logging
_logger = logging.getLogger(__name__)
# 3rd party modules
from dask.distributed import Client
# custom modules
from generator import *


if __name__ == '__main__':
    logging.basicConfig(
        format = '%(asctime)s %(name)s %(levelname)s: %(message)s',
        level = logging.INFO,
    )
    logging.getLogger('__main__').setLevel(logging.INFO)
    client = Client(
        dashboard_address = ':8605',
        n_workers = 12,
        threads_per_worker = 1,
    )
    _logger.info(f'dask distributed scheduler link: {client.dashboard_link}')
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret,dict):
        globals().update(cli_ret)
