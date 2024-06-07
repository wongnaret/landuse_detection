"""Load multiple npz, perform crop_age shift on each, then merge them.

The resulting dataset will be stored in variable `ds` accessible when running python interactively.
"""
from pathlib import Path
import yaml
import xarray as xr
import click
# custom modules
from shapeshift_npz import load_shifted


prefix = Path('../generator/data/class0-outside-farm')
prep_opt = 'mask_rmmeh' # Alternatively, 'og' and 'unmask_rmmeh' could be used.
default_configs = yaml.safe_load("""
'2019':
    - - 2019/bk_{}_full_data
      - 2019-01
      - 2019-04
    - - 2019/dc_{}_full_data
      - 2018-12
      - 2019-02
    - - 2019/kc_{}_full_data
      - 2018-12
      - 2019-03
'2020':
    - - 2020/bk_{}_full_data
      - 2019-12
      - 2020-05
    - - 2020/dc_{}_full_data
      - 2019-12
      - 2020-02
    - - 2020/kc_{}_full_data
      - 2019-12
      - 2020-03
""")
for ls in default_configs.values():
    for l in ls:
        l[0] = prefix.joinpath(l[0].format(prep_opt))
default_configs['2019&2020'] = default_configs['2019'] + default_configs['2020']
default_configs = {
    '{}-{}'.format(k,s): [
        (conf[0].joinpath(s + 'Set.npz'), conf[1], conf[2])
        for conf in configs
    ]
    for k, configs in default_configs.items()
    for s in ('train','test')
}


# https://github.com/pallets/click/issues/405#issuecomment-470812067
class ClickPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


def load_shift_merge(configs):
    """Load npz files, change time axis to crop_age and combine them all.
    
    load_shifted_configs -- list(tuple(file,str,str))
        List of 3-tuple of arguments for shapeshift_npz.load_shifted
    
    return -- xarray.DataArray
        being the concatenation along 'location' dimension of input datasets
        with new coordinate 'merge_source' being the index of the source dataset as given in 'configs' list.
    """
    return xr.concat(
        (
            load_shifted(*conf)[1].assign_coords(merge_source=i)
            for i,conf in enumerate(configs)
        ),
        'location',
    )

"""# The follow tests that concat along 'location' really does union on 'crop_age'
s = [
    load_shifted(*conf)[1].assign_coords(merge_source=i)
    for i,conf in enumerate(configs)
]
ss = [
    s[0].sel(crop_age=slice('10 days')),
    s[1].sel(crop_age=slice('0 days', None)),
]
c = xr.concat(ss,'location')
c.isel(location=c.merge_source==0).dropna('crop_age','all')
c.isel(location=c.merge_source==1).dropna('crop_age','all')
"""

@click.group(
    invoke_without_command = True,
    help = __doc__,
)
@click.option(
    '--preset',
    type = click.Choice(default_configs.keys()),
    show_default = True,
    help = 'Combine from a preset source datasets.',
)
@click.option(
    '--save-nc',
    type = ClickPath(dir_okay = False),
    help = 'Save resulting dataset `ds` to this path.',
)
@click.option(
    '--src',
    type = (ClickPath(exists = True, dir_okay = False), str, str),
    multiple = True,
    help = 'A triple describing (source file, begin date of crop-start search, end date of crop-start search). '
        'Giving this option multiple times will merge the shifted datasets into one. '
        'If --preset is also given, --src will add to the preset.',
)
def cli(preset, save_nc, src):
    configs = [
        *default_configs.get(preset, tuple()),
        *src,
    ]
    if len(configs) > 0:
        ds = load_shift_merge(configs)
        if save_nc is not None:
            save_nc.parent.mkdir(parents = True, exist_ok = True)
            ds.reset_index('location').to_netcdf(save_nc)
    return locals()


# unit test on default configs
if __name__ == '__main__':
    cli_ret = cli(standalone_mode = False)
    if isinstance(cli_ret,dict):
        globals().update(cli_ret)
