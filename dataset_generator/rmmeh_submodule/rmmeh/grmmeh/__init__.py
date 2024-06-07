"""Smooth timeseries in ND-arrays using RMMEH."""
import scipy.signal
import numpy as np
import pandas as pd
import xarray as xr


_default_hanning_window = 5
_default_conv_method = 'direct'


def nd_avg_neighbor(a,dim):
    """Take average of neighboring values along given dimension.
    
    This will modifies the array in-place.
    The edges where neighbor is missing, the result will be np.nan.
    This means dtype of original array needs to support np.nan.
    """
    # initialise indexer & pad_width to be all empty
    prev_indexer = [slice(None)] * len(a.shape)
    next_indexer = prev_indexer.copy()
    dst_indexer = prev_indexer.copy()
    bpad_indexer = prev_indexer.copy()
    fpad_indexer = prev_indexer.copy()
    # specify the shift in the indexer
    prev_indexer[dim] = slice(None,-2)
    next_indexer[dim] = slice(2,None)
    dst_indexer[dim] = slice(1,-1)
    bpad_indexer[dim] = slice(None,1)
    fpad_indexer[dim] = slice(-1,None)
    # reusable location of 'a'
    dst = a[tuple(dst_indexer)]
    # calculation
    np.add(a[tuple(prev_indexer)], a[tuple(next_indexer)], dst)
    np.multiply(dst, .5, dst)
    a[tuple(bpad_indexer)] = np.nan
    a[tuple(fpad_indexer)] = np.nan
    return a


def linear_fill(a,dim, max_gap = None):
    # TODO xarray has overhead, could be faster if stayed within pure numpy
    da = xr.DataArray(a)
    da = da.assign_coords({
        da.dims[dim]: pd.RangeIndex(0, a.shape[dim])
    })
    return da.interpolate_na(
        da.dims[dim],
        use_coordinate = False,
        max_gap = max_gap,
    ).values


def nd_rmmeh(
    a, dim,
    median_window = 5, hanning_window = _default_hanning_window,
    conv_method = _default_conv_method, use_max = True,
    details = False,
    linear_fill_max_gap = 5,
):
    """N-dimensional version of RMMEH, optimised for performance on numpy.ndarray.
    
    a -- numpy.ndarray
        Source array to be smoothed.
    dim -- non-negative integer
        The index of time dimension to smooth values along.
    median_window -- positive odd integer
        The size of window used for calculation of running median.
    hanning_window -- positive odd integer
        The size of hanning window used to do the final weighted moving average.
    conv_method -- str {‘auto’, ‘direct’, ‘fft’}, optional
        The convolution method used to produce hanning moving average.
        See scipy.signal.convolve for more details.
    use_max -- Boolean
        Use maximum to conbine (original, running medians,average neighbor).
        Otherwise, use minimum.
    details -- Boolean
        Whether to also output intermediate values.
        Useful for debugging and reusing intermediate values.
        If True will output three arrays `(rmmeh,maxed,running medians,average neighbor)` instead of just `rmmeh`.
    linear_fill_max_gap -- non-negative integer or None
        The maximum length of NaN gap that will be filled with linear interpolation as preprocessing.
        The length of a gap is calculated by difference in indexes of valid data on both sides.
        Therefore, a gap of length `n` contains `n-1` consecutive NaNs.
        Setting value to 1 or less effectively does not fill any NaN.
        Setting value to None removes the limit, causing internal NaNs to always be interpolated.
    """
    if (
        median_window < 1 or median_window % 2 != 1 or
        hanning_window < 1 or hanning_window % 2 != 1
    ):
        raise ValueError('window size must be positive odd integer')
    decider = np.fmax if use_max else np.fmin
    
    ## linear interpolation is optionally applied on small gaps
    # this helps reduces the number of resulting NaNs during intermediate steps
    
    if linear_fill_max_gap is None or linear_fill_max_gap > 1:
        a = linear_fill(a, dim, max_gap = linear_fill_max_gap)
    
    ## construct median filtered version of 'a' then immediately decide whether to keep
    
    median_kernal = np.full(len(a.shape),1)
    median_kernal[dim] = median_window
    maxed = scipy.signal.medfilt(
        a, median_kernal
    )
    if details:
        medians = maxed.copy()
    decider(a,maxed,maxed)
    
    ## construct average of neighbors, then immeidately decide whether to keep
    
    an = nd_avg_neighbor(a.copy(),dim)
    decider(an,maxed,maxed)
    if not details:
        del an
    
    ## final hanning moving average
    
    rmmeh_result = nanhann(a, dim, hanning_window, conv_method)
    
    if details:
        return rmmeh_result, maxed, medians, an
    else:
        return rmmeh_result


def nanhann(
    a, dim,
    hanning_window = _default_hanning_window,
    conv_method = _default_conv_method,
):
    if hanning_window <= 3:
        return a
    dim_complement = tuple(x for x in range(len(a.shape)) if x != dim)
    hanning_kernel = np.expand_dims(
        np.hanning(hanning_window),
        axis = dim_complement,
    )
    trim_indexer = [slice(None)] * len(a.shape)
    trim_indexer[dim] = slice(hanning_window//2, -(hanning_window//2))
    trim_indexer = tuple(trim_indexer)
    rmmeh_result = scipy.signal.convolve(
        np.nan_to_num(a, nan = 0),
        hanning_kernel,
        method = conv_method,
    )[trim_indexer]
    normaliser = scipy.signal.convolve(
        ~np.isnan(a),
        hanning_kernel,
        method = conv_method,
    )[trim_indexer]
    np.true_divide(rmmeh_result, normaliser, rmmeh_result)
    return rmmeh_result

