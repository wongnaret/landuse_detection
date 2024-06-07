import matplotlib.pyplot as plt
from kea.broker.probav.adhoc_reader import load as probav_load


def plot_probav_at_location(lat, lon):
    pvds = probav_load(date_slice = slice('2019','2020'))
    pvds = pvds.sel(
        latitude = slice(lat + 0.003, lat - 0.003),
        longitude = slice(lon - 0.003, lon + 0.003),
    ).load()
    #
    plt.figure(figsize = (12,10))
    for ilat in range(pvds.sizes['latitude']):
        for ilon in range(pvds.sizes['longitude']):
            pvds[:,ilat,ilon].plot(alpha = .5)
    plt.savefig('img/probav_lai_around_{:.3f}_{:.3f}.png'.format(lat,lon))
    plt.close()
    return pvds


"""# example
pvds = plot_probav_at_location(16.459, 102.097)
pvds = plot_probav_at_location(16.297, 102.185)
"""
