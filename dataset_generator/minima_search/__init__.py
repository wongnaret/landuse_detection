def minima_search(
    da, radius,
    dim = 'time',
    min_periods = None,
    tiebreak = None,
):
    """Mark locations which has the lowest value compared to all its neighbors within a radius.
    
    Parameters
    ----------
    da: xarray.DataArray
        The data
    dim: str
        dimension name to search for minimas along
    Radius: int
    min_periods: int or None
        Number of non-NaN observations required in radius
        (otherwise the position is automatically considered as non-minima).
        See also, xarray.DataArray.rolling.
    tiebreak: None, 'low', or 'high'
        The strategy in case there are many equal local minima closer than radius.
        'low' or 'high' takes the minimas from the lowest or highest index respectively.
    
    Returns
    -------
    is_minima: xarray.DataArray of Boolean
        DataArray with same coordinates as input `da`.
    """
    # TODO: adapting the following strategy might speed up
    # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/_peak_finding.py#L22
    is_minima = da.rolling(
        {dim: radius*2 + 1},
        min_periods = min_periods,
        center = True,
    ).min() == da
    
    # optionally tiebreak minimas
    if tiebreak is not None:
        flip_indexer = {dim: slice(None,None,-1)}
        if tiebreak == 'high':
            t = is_minima.isel(flip_indexer)
        elif tiebreak != 'low':
            raise ValueError("Tiebreak must be None, 'low', or 'high'")
        t = t.rolling({dim: radius + 1}, min_periods = 1).sum()
        if tiebreak == 'high':
            t = t.isel(flip_indexer)
        is_minima = (t == 1) & is_minima
        del t, flip_indexer
    
    is_minima.name = 'is_minima'
    is_minima.attrs['radius'] = radius
    is_minima.attrs['minima_along'] = dim
    return is_minima
