"""Unit tester for RMMEH."""
import matplotlib.pyplot as plt
import pandas as pd
# custom modules
from . import *
from ..signal_sample import stock_price


if __name__ == '__main__':
    ts = stock_price(dates=pd.date_range('2020-01-01', '2020-02-01'))
    # NOTE: don't set details = True if you only need the result RMMEH
    rets = rmmeh(ts,details = True)
    # visualise
    alpha = 0.25
    plt.figure(figsize=(13,8))
    for x in reversed(rets[1:]):
        x.plot(alpha=alpha,marker='x',linestyle='dashed')
    ts.plot()
    rets[0].plot()
    plt.title('RMMEH and intermediate values.')
    plt.legend()
    plt.show()

