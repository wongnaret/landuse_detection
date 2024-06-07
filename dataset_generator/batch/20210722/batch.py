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
    d.pop(2019,None)
    d.pop(2020,None)
    d.pop(2021,None)

# where the generator script is
dir_gen = Path(__file__).parent.joinpath('../../generator/')
# where plotting script is
dir_plt = Path(__file__).parent.joinpath('../../shift/')
# TODO make this dynamic
dir_gen_relto_plt = Path('../generator')


# part of command that stays the same
cmd_base = [
    'time', 'python', '__main__.py', '--no-preview', '--fill-missing-week', '--class-shp',
    '/home/wongnaret/repositories/lstm/lstm_dataset/20210714_dataset/shp/mitrphol_data_for_LSTM.shp',
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
            outdir = 'data/{}/{}_{}_full_data'.format(year, farm.lower(), option)
            dir_opt_data[option] = outdir
            cmd = [ *cmd_noopt, '-o', outdir, *optargs ]
            print(' '.join(cmd))
            if not dry_run:
                subprocess.run(cmd, cwd = str(dir_gen))
        
        ## once all options are done, produce plot
        
        cmd_plot = [
            'time', 'python', 'merge_shift_plot.py',
            '--plot-y-begin', '-.5', '--plot-y-end', '1',
            *itertools.chain.from_iterable(
                (
                    '--npz',
                    farm + opt2display[option],
                    str(dir_gen_relto_plt.joinpath(indir,'trainSet.npz')),
                )
                for option, indir in dir_opt_data.items()
            ),
            '-b', begin, '-e', end,
            '-o', 'output/aligned/{}/{}'.format(year, farm.lower()),
        ]
        print(' '.join(cmd_plot))
        if not dry_run:
            # TODO this could be Popen in background instead, but you need somehow chain the two subprocesses
            subprocess.run(cmd_plot, cwd = dir_plt)
        
        ## recombine images
        cmd_recom = [ 'bash', 'recombine.sh', 'output/aligned/{}/{}'.format(year, farm.lower()) ]
        print(' '.join(cmd_recom))
        if not dry_run:
            subprocess.run(cmd_recom, cwd = dir_plt)
