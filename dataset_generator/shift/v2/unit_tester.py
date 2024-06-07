from pathlib import Path
# 3rd party modules
import xarray as xr
# custom modules
from shift import *


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s %(message)s',
    )
    example_file = next(
        #Path('/srv/data-dir/storage/lstm_dataset/20211012_dataset').glob('*/*.nc')
        Path('/srv/data-dir/storage/lstm_dataset/20211012_dataset/kpp1_rmmeh_masked').glob('*.nc')
        #Path('/srv/data-dir/storage/lstm_dataset/20211122_dataset/kpp1_rmmeh_masked').glob('*.nc')
        #Path('/srv/data-dir/storage/lstm_dataset/20211012_dataset/north_eastern1_rmmeh_masked').glob('*.nc')
    )
    dss = DatasetSlicer(example_file)
    aged_batch = dss.get_batch(
        label = 1, # can also be list of integers such as [1,2]
        ntimestep = 53, # There are 53 timesteps starting from age = 0 weeks to age = 52 weeks
        nlocation = 200,
        
        # Age of plant at the ending of profile (in number of time steps).
        # By default, this is `ntimestep - 1`.
        # age = 52,
    )
    random_batch = dss.get_random_slice_batch(
        label = [0, 2], # can also be single integer such as 0
        ntimestep = 36,
        nlocation = 200,
    )
    logging.info('Finish unit tester')
