def plant_mask(ds, dim = 'time', low = .1, high = 1.):
    """Produces 'plant' mask as a coordinate based on NDVI.
    
    NDVI will be retrived from data variable with name 'ndvi' or calculated from 'red' and 'nir' data variables.
    
    dim: str or list of str
        Name(s) of the dimensions to aggregate along.
    low, high: floats
        Inclusive range of NDVI values considered to be plant.
    """
    ds = ds.copy()
    ndvi = ds.data_vars.get(
        'ndvi',
        (ds.nir - ds.red) / (ds.nir + ds.red)
    ).median(dim)
    return (low <= ndvi) & (ndvi <= high)
