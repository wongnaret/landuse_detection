"""Applying gRMMEH on xarray objects.

- Helps perform gRMMEH in chunks.
"""
import numpy as np
# custom modules
from . import nd_rmmeh
from ..xarray_utils import autochunk


# TODO: handle resampling
def apply_rmmeh(da, *, dim = 'time', **kwargs):
    """Apply gRMMEH smoothing on xarray.DataArray in chunks.
    
    obj: xarray.DataArray
        The object to perform gRMMEH on.
    dim: str
        name of the dimension to smooth along.
    kwargs:
        extra key word arguments passed to `nd_rmmeh`.
    """
    da = autochunk.chunk(da, chunks = {dim: -1})
    return da.copy(
        deep = False,
        data = da.data.map_blocks(
            nd_rmmeh,
            dtype = np.float,
            meta = da.data,
            # the following is kwargs to nd_rmmeh
            dim = da.get_axis_num(dim),
            **kwargs,
        ),
    )
