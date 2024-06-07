import sys
import datetime
from pathlib import Path


def cmd_log(outdir):
    """Log time and command line argument used as 'cmd' file in given directory"""
    outdir = Path(outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    with open(outdir.joinpath('cmd'),'a') as f:
        print('===', datetime.datetime.now(), file = f)
        print(sys.argv, file = f)
        print(' '.join(sys.argv), file = f)
