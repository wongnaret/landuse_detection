from pathlib import Path
# 3rd party
import numpy as np
import xarray as xr
# custom
from plant_mask import plant_mask


example_file = next(Path('/srv/data-dir/storage/lstm_dataset/20211012_dataset').glob('*/*.nc'))
print('Opening', example_file)
ds = xr.open_dataset(example_file)

pnp = plant_mask(ds)
def nonzero_percent(da):
    return (np.count_nonzero(da) * 100) / np.prod(da.shape)
print('percentage of locations detected as plant:', nonzero_percent(pnp))