"""Unit tester for minima_search"""
import xarray as xr
# custom modules
from . import *
from ..rmmeh.test.generate_example import gen_ex


if __name__ == '__main__':
    globals().update(gen_ex(na_chance = 0))
    a = xr.DataArray(a)
    r = minima_search(a, 3, dim = a.dims[dim])
