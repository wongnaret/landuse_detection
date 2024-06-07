#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: conf.py
 Date Create: 14/9/2021 AD 08:58
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

import numpy as np

ID_COLS = ['series_id', 'measurement_number']

'''
['coastal', 'blue', 'green', 'red', 'veg5', 'veg6', 'veg7', 'nir',
       'narrow_nir', 'water_vapour', 'swir1', 'swir2', 'SCL', 'WVP',
       'AOT']
'''
x_cols = {
    'series_id': np.uint32,
    'measurement_number': np.uint32,
    'ndvi': np.float32,
    'ndwi': np.float32,
    'coastal': np.float32,
    'blue': np.float32,
    'green': np.float32,
    'red': np.float32,
    'veg5': np.float32,
    'veg6': np.float32,
    'veg7': np.float32,
    'nir': np.float32,
    'narrow_nir': np.float32,
    'water_vapour': np.float32,
    'swir1': np.float32,
    'swir2': np.float32,
    'SCL': np.float32,
    'WVP': np.float32,
    'AOT': np.float32
}

y_cols = {
    'series_id': np.uint32,
    'class':  np.uint32,
}