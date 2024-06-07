from . import *
from ..test.generate_example import xr_gen_ex


if __name__ == '__main__':
    globals().update(xr_gen_ex(shape = (2,10,2), dim = 1, chunks = -1))
    an = avg_neighbor(a, 'time')
    hann = nanhann(a, 'time', hanning_window = 11)
    result = rmmeh(a, 'time', hanning_window = 11)
