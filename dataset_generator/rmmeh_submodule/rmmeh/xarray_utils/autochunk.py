"""Automatically pick chunk size in unspecified dimension.

Use case:
    Want to specify chunk only in some dimensions while
    letting the rest of dimensions (which is unknown) defaults to 'auto'.

Dask normally only defaults chunk specification to 'None' instead of 'auto'.
This module gives a convenient way to override the default choice.
It provides functions with the same name as standard xarray functions,
but `chunks` arguments is captured and modified before passing to the standard functions.
"""
import xarray as xr


def chunks_spec(obj, chunks = None, default_chunk_size = 'auto', **kwargs):
    """Make chunks specification for xarray object `obj`
    
    such that dimensions not specified in `chunks` or `kwargs` gets chunks size `default`.
    """
    return {
        dim: chunks.get(
            dim,
            kwargs.get(dim, default_chunk_size)
        )
        for dim in obj.dims
    }


def open_dataset(filename_or_obj, *, chunks = None, default_chunk_size = 'auto', **kwargs):
    """Like xarray.open_dataset but chunks of unspecified dimensions can be set to an arbitrary default."""
    with xr.open_dataset(
        filename_or_obj,
        decode_cf = False,
        decode_times = False,
        decode_timedelta = False,
        decode_coords = False,
    ) as ds:
        chunks_ = chunks_spec(ds, chunks, default_chunk_size)
    return xr.open_dataset(filename_or_obj, chunks = chunks_, **kwargs)


def chunk(obj, chunks = None, default_chunk_size = 'auto', **kwargs):
    """Like xarray.DataArray.chunk or xarray.Dataset.chunk, but unspecified dimensions can be set to an arbitrary default."""
    return obj.chunk(
        chunks_spec(obj, chunks, default_chunk_size), **kwargs
    )


# unit tester
if __name__ == '__main__':
    example_file = '/srv/data-dir/storage/lstm_dataset/20211012_dataset/kpp1_rmmeh_masked/trainSet.nc'
    ds = open_dataset(
        example_file,
        chunks = dict(time = -1),
    )
    ds_ = chunk(ds, chunks = dict(location = 2000000))
