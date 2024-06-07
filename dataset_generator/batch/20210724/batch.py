import yaml
import subprocess
from pathlib import Path
import datetime
import pandas as pd
from pprint import pprint
import itertools

dry_run = False

with open('task.yaml','r') as f:
    tasks = yaml.safe_load(f)

# ignore some years
for farm,d in tasks.items():
    d.pop(2018,None)
    d.pop(2021,None)

# where the generator script is
dir_gen = Path(__file__).parent.joinpath('../../generator/')
# where plotting script is
dir_plt = Path(__file__).parent.joinpath('../../shift/')
# TODO make this dynamic
dir_gen_relto_plt = Path('../generator')


# part of command that stays the same
cmd_base = [
    'time', 'python', '__main__.py', '--no-preview', '--fill-missing-week',
    '--class-shp', '/home/wongnaret/repositories/lstm/lstm_dataset/20210722_dataset/shp/non_sugar_lstm.shp',
    '--class-shp', '/home/wongnaret/repositories/lstm/lstm_dataset/20210714_dataset/shp/mitrphol_data_for_LSTM.shp',
    '--map-label', '0', '-1',
    '--map-label', '1', '0',
    '--map-label', '2', '1',
    '--n-sample', '15000',
]

# spatial extent mapping
se_map = {
    'BK': ['--latitude-extent', '16.3175', '16.2907', '--longitude-extent', '102.1741', '102.2040'],
    'DC': ['--latitude-extent', '14.9092', '14.8559', '--longitude-extent', '99.7549', '99.8402'],
    'KC': ['--latitude-extent', '16.5019', '16.4301', '--longitude-extent', '102.0488', '102.1398'],
}

options = {
    'og': [],
    'unmask_rmmeh': ['--rmmeh', '--rmmeh-hanning-window', '11'],
    'mask_rmmeh': ['--rmmeh', '--rmmeh-hanning-window', '11', '--mask-scl'],
}

opt2display = {
    'og': '-original',
    'unmask_rmmeh': '-unmasked-rmmeh',
    'mask_rmmeh': '-masked-rmmeh',
}

# main generation
for farm,d in tasks.items():
    for year,(begin,end) in d.items():
        print(datetime.datetime.now(),'generating data for',farm,year,begin,end)
        cmd_noopt = [
            *cmd_base, *se_map[farm],
            '--time-extent',
            str((pd.to_datetime(begin) - pd.offsets.DateOffset(months=1)).date()),
            str((pd.to_datetime(end) + pd.offsets.DateOffset(years=1)).date()),
        ]
        
        ## generate datasets for each options
        
        dir_opt_data = {}
        for option, optargs in options.items():
            outdir = 'data/class0-outside-agrimap/{}/{}_{}_full_data'.format(year, farm.lower(), option)
            dir_opt_data[option] = outdir
            cmd = [ *cmd_noopt, '-o', outdir, *optargs ]
            print(' '.join(cmd))
            if not dry_run:
                subprocess.run(cmd, cwd = str(dir_gen))
