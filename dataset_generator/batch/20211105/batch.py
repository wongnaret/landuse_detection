import subprocess
from pathlib import Path
# 3rd party modules
import yaml


dry_run = True

# where the generator script is
dir_gen = Path(__file__).parent.joinpath('../../generator/')

# base command to generate datasets
cmd_base = [
    'time', 'python', '__main__.py', '--fill-missing-week',
]

# get description of all datasets
with open(Path(__file__).parent.joinpath('datasets.yaml'),'r') as f:
    datasets = yaml.safe_load(f)

# append with settings common accross datasets
## class description
cmd_base += [
    y
    for x in datasets['common']['class']
    for y in ('--class-shp', x['path'])
]
## time extent
cmd_base += [
    '--time-extent', *datasets['common']['time']
]


for ds in datasets['datasets']:
    cmd = [
        *cmd_base,
        '--latitude-extent', *map(str, ds['latitude']),
        '--longitude-extent', *map(str, ds['longitude']),
        '-o', ds['name'],
    ]
    print(' '.join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd = str(dir_gen))
