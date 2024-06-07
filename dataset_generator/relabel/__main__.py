import logging
_logger = logging.getLogger(__name__)
# custom modules
from .relabel import cli


if __name__ == '__main__':
    logging.basicConfig(
        format = '%(asctime)s %(name)s %(levelname)s: %(message)s',
        level = logging.INFO,
    )
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret, dict):
        globals().update(cli_ret)
