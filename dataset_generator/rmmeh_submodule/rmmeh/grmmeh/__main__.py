"""Unit tester of gRMMEH."""
from . import *
from ..test.generate_example import gen_ex


# unit tester
if __name__ == '__main__':
    globals().update(gen_ex())
    
    # call test subject
    rmmeh_result, maxed, medians, an = nd_rmmeh(
        a, dim,
        median_window = 5,
        hanning_window = 7,
        use_max = False,
        details = True,
        linear_fill_max_gap = 5,
    )
    # TODO plot the result

