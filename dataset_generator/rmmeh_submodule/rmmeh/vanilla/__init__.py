"""Smooth timeseries using RMMEH.

RMMEH is a timeseries smoothing method originally proposed in
[A novel compound smoother - RMMEH to reconstruct MODIS NDVI time series]
(https://www.researchgate.net/publication/258789353_A_novel_compound_smoother_-_RMMEH_to_reconstruct_MODIS_NDVI_time_series).
The original target singal of the paper was the NDVI from MODIS.
The signal needs to be smoothed in order to remove errorneous noise.

This methods makes two important assumption:
1. The true signal is smooth and gradual in change.
2. The error only cause decrease in signal value, never an increase.
"""
import pandas as pd


def avg_neighbor(ts):
    out = (ts[:-2].reset_index(drop=True) + ts[2:].reset_index(drop=True))/2
    out.index = ts.index[1:-1]
    return out

def rmmeh(ts,median_window=5,details=False):
    """Original version of RMMEH, with more limitations on input.
    
    ts -- pandas.Series
        Must have regular frequency, without missing values.
    median_window -- positive odd integer
        The size of window used for calculation of running median.
    details -- Boolean
        Whether to also output intermediate values.
        Useful for debugging and reusing intermediate values.
        If True will output three series `(rmmeh,maxed,running medians,average neighbor)` instead of just `rmmeh`.
    """
    medians = ts.rolling(
        window = median_window,
        center = True,
        min_periods = 1,
    ).median()
    an = avg_neighbor(ts)
    maxed = pd.DataFrame([ts,medians,an]).max()
    # hann weighted moving average with weights 0,.25,.5.,.25,0
    rmmeh_result = maxed.rolling(
        window = 5,
        win_type='hann',
        center=True,
        min_periods=1
    ).mean()
    rmmeh_result.name = 'RMMEH of '+ts.name
    if details:
        maxed.name = 'max of 3 RMMEH components'
        medians.name = 'running median of '+ts.name
        an.name = 'average neighbor of '+ts.name
        return rmmeh_result, maxed, medians, an
    else:
        return rmmeh_result
