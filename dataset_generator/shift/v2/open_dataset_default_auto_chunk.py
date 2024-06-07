import xarray as xr


def open_dataset_default_auto_chunk(*args, chunks = None, **kwargs):
    """Like xarray.open_dataset but dimensions not mentioned in `chunks` gets 'auto' instead of 'None'.
    
    chunks: dict
        Chunk specification. Only support str-key dictionary format.
    """
    with xr.open_dataset(*args, chunks = -1, **kwargs) as ds:
        chunks_ = {
            x: chunks.get(x, 'auto')
            for x in ds.dims
        }
    return xr.open_dataset(*args, chunks = chunks_, **kwargs)
