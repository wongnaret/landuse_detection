"""Test gRMMEH on random sequence"""
import numpy as np
import xarray as xr
# custom modules
from ..nd_rmmeh import nd_rmmeh


def rand_seq(
    length = 100,
    na_chance = .5, low = -1., high = 1.,
    median_window = 5, hanning_window = 11,
):
    a = np.random.uniform(low = low, high = high, size = length)
    a[np.random.choice(length, int(na_chance*length), replace = False)] = np.nan
    # call test subject
    rmmeh_result, maxed, medians, an = nd_rmmeh(
        a, 0,
        median_window = median_window,
        hanning_window = hanning_window,
        use_max = False,
        details = True,
    )
    # check correctness
    if ((rmmeh_result > high) | (rmmeh_result < low)).any():
        return False, locals()
    return True, locals()


"""
for i in range(100):
    for na_chance in (.5,.7,.9,1.):
        result, loc = rand_seq(na_chance = na_chance)
        if not result:
            break
"""


def test_nd(
    size = (100,100),
    na_chance = .5, low = -1., high = 1.,
    median_window = 5, hanning_window = 11,
):
    a = np.random.uniform(low = low, high = high, size = size)
    a[np.random.rand(*size) > na_chance] = np.nan
    ret = nd_rmmeh(
        a, 0,
        median_window = median_window,
        hanning_window = hanning_window,
        use_max = False,
        conv_method = 'direct',
    )
    return a, ret


"""
for width in range(2,20):
    a, r = test_nd(size = (100,width))
    err = ((r < -1) | (r > 1))
    if err.any():
        print(width)
        print(np.nonzero(err))
        break


# once the input dimensions are large enough, the same place which should output NaN always acts up.
t = nd_rmmeh(
    a[:,:-1], 0,
    median_window = 5, hanning_window = 11,
    use_max = False,
)
np.nonzero((t < -1) | (t > 1))

# 
t, maxed, medians, an = nd_rmmeh(
    a, 0,
    median_window = 5, hanning_window = 11,
    use_max = False,
    details = True,
    conv_method = 'direct',
)
np.nonzero((t < -1) | (t > 1))
an[56:,1]
medians[56:,1]
maxed[56:,1]
"""


def test_xr_rmmeh(
    size = (100,100),
    na_chance = .5, low = -1., high = 1.,
    median_window = 5, hanning_window = 11,
):
    #Test the chunked application of nd_rmmeh
    da = xr.DataArray(
        np.random.uniform(low = low, high = high, size = size),
    )
    da = da.where(np.random.rand(*size) > na_chance)
    dac = da.chunk(
        [da.shape[0]] + 
        [x//5 for x in da.shape[1:]]
    )
    chunk_ret = dac.copy(
        deep = False,
        data = dac.data.map_blocks(
            nd_rmmeh,
            dtype = dac.dtype,
            meta = dac.data,
            # the following is kwargs to nd_rmmeh
            dim = 0,
            median_window = median_window,
            hanning_window = hanning_window,
            use_max = False,
        )
    )
    norm_ret = nd_rmmeh(
        da.values, 0,
        median_window = median_window,
        hanning_window = hanning_window,
        use_max = False,
    )
    return da, chunk_ret, norm_ret


"""
da, r, nr = test_xr_rmmeh()
r.load()
r.where((r < -1) | (r > 1), drop = True)
da.where((r < -1) | (r > 1), drop = True)
xr.DataArray(nr).where((lambda x: (x < -1) | (x > 1)), drop = True)
"""
