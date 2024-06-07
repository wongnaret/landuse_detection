"""Unfinished alternative to finding neighbor average."""
import numpy as np


def nd_interp_neighbor(a,dim):
    """Interpolate between the two closest non-nan neighbors on both sides within a window size along a given dimension.
    
    This approach is abandoned in favour of the simpler (to implement)
    "linear_fill as preprocessing to nd_avg_neighbor".
    New approach also does not have to worry about window size.
    
    This method has not implemented limiting window size yet.
    
    This method modifies array in-place.
    
    Example of how how vectorized calculation below works:
    0123456 location
    7..59.2 value
    x..xx.x valid
    0003446 ff
    0333466 bf
     00034  ln = ff[:-2]
     33466  rn = bf[2:]
    
    reference:
        https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    import warnings
    warnings.warn(
        "This approach is abandoned in favour of the simpler (to implement) "
        "'linear_fill as preprocessing to nd_avg_neighbor'.",
        DeprecationWarning
    )
    
    empty_indexer = [slice(None)] * len(a.shape)
    
    dim_complement = tuple(x for x in range(len(a.shape)) if x != dim)
    loc = np.expand_dims(
        np.arange(a.shape[dim]),
        axis = dim_complement,
    )
    
    # location of non-nan left and right neighbors
    # can be read from back and front fill source location respectively
    anotnan = ~np.isnan(a)
    
    ff = np.where(anotnan, loc, 0)
    np.maximum.accumulate(ff, axis = dim, out = ff)
    
    bf = np.where(anotnan, loc, a.shape[dim] - 1)
    fbf = np.flip(bf)
    np.minimum.accumulate(fbf, axis = dim, out = fbf)
    
    del anotnan, fbf
    
    # leave out the boundary which can not have both neighbors, they will be padded with nan later
    # now we get neighbors index for ones in the middle
    discard_trail_indexer = empty_indexer.copy()
    discard_trail_indexer[dim] = slice(None,-2)
    ln = ff[tuple(discard_trail_indexer)]
    
    discard_lead_indexer = empty_indexer.copy()
    discard_lead_indexer[dim] = slice(2,None)
    rn = bf[tuple(discard_lead_indexer)]
    
    # get left and right neighbor's value (lnv,rnv)
    broadcast_indexer = [
        np.expand_dims(
            np.arange(a.shape[i]),
            tuple(x for x in range(len(a.shape)) if x != i)
        )
        for i in range(len(a.shape))
    ]
    broadcast_indexer[dim] = ln
    lnv = a[tuple(broadcast_indexer)]
    broadcast_indexer[dim] = rn
    rnv = a[tuple(broadcast_indexer)]
    
    # current index in contrast to ln and rn
    discard_both_ends_indexer = empty_indexer.copy()
    discard_both_ends_indexer[dim] = slice(1,-1)
    c = loc[tuple(discard_both_ends_indexer)]
    
    # interpolate between neighbors
    a[tuple(discard_both_ends_indexer)] = ((rn-c)*lnv + (c-ln)*rnv)/(rn-ln)
    
    # pad with nan at borders
    bpad_indexer = empty_indexer.copy()
    fpad_indexer = empty_indexer.copy()
    bpad_indexer[dim] = slice(None,1)
    fpad_indexer[dim] = slice(-1,None)
    a[tuple(bpad_indexer)] = np.nan
    a[tuple(fpad_indexer)] = np.nan
    
    return a
