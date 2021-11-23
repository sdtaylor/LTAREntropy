import datetime
import json
import os
from os import path
import time

import pandas as pd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box

from tools.geotools import transform_shape_crs

from tqdm import tqdm

all_scene_combos = [
    dict(
        ltar_domain = 'umrb',
        raster_files = dict(
            planet = './data/imagery/f4e1808f-4a56-4ace-a3f7-1330f6216825/PSScene4Band/20200718_164445_64_106c_3B_AnalyticMS_SR_bandmath.tif',
            l8     = './data/imagery/umrb_L8_NDVI.tif',
            s2     = 'data/imagery/umrb_S2_NDVI.tif'
            )

        ),
    dict(
        ltar_domain = 'jer',
        raster_files = dict(
            planet = 'data/imagery/c8ed0365-3487-4522-8a96-4a0c43f10a63/PSScene4Band/20200718_174728_35_1058_3B_AnalyticMS_SR_bandmath.tif',
            l8     = 'data/imagery/jer_L8_NDVI.tif',
            s2     = 'data/imagery/jer_S2_NDVI.tif'
            )

        ),
    ]


n_random_samples=1000
window_sizes = [500,1000] # in meters
entropy_results_file = './data/entropy_results.csv'
ndvi_sample_file = './data/scene_ndvi_samples.csv'

sensor_pixel_size = dict(
    planet=3,
    l8=30,
    s2=10
    )

#----------------------------------------
# Helper Functions


def box_from_midpoint_and_size(x, y, size):
    """All in meters"""
    offset = size/2
    return box(
        minx = x - offset,
        miny = y - offset,
        maxx = x + offset,
        maxy = y + offset
        )

def differential_entropy(arr):
    """ 
    Differential entropy given a normal distribution
    #https://en.wikipedia.org/wiki/Differential_entropy
    """
    constant = np.sqrt(np.e * np.pi * 2)
    return np.log(np.std(arr) * constant)

def entropy_calculation(data, pixel_size):
    """
    The core entropy calculation with the following steps:
    1. round NDVI values to 2 decimals places, convert to integers for easier use.
        This results in 201 possible values, from -100 - 100
    2. Calculate total area of each unique NDVI value.
    3. Calculate shannon entropy based on aereal proportion of each unique one.

    Parameters
    ----------
    data : np.array
        A 2d array of ndvi values in the  range -1-1.
    pixel_size : int
        size in meters of eac pixel in data array

    Returns
    -------
    None.

    """
    rounded = (data * 100).round(0).astype(np.int16)
    # with return_counts=True a tuple is returned, w/ element 1 as the 
    # counts of each unique element.
    # Thus sample size N is the length of counts
    counts = np.unique(rounded, return_counts=True)[1] 
    areas = counts * (pixel_size**2)
    areas = areas / areas.sum() # proportions
    return -np.sum(areas * np.log(areas)) # np.log default is natural log.

#----------------------------------------
# Primary loops


all_results = []

for scene_combo in all_scene_combos:
    pass
    # Preload everything into memory
    ndvi_data = {}
    for sensor, raster_file in scene_combo['raster_files'].items():
        ndvi_data[sensor] = {}
        with rasterio.open(raster_file) as src:
            ndvi_data[sensor]['data'] = src.read()
            ndvi_data[sensor]['profile'] = src.profile
            
            if sensor=='planet':
                # TODO: Naively set planet nodata until I can do it with the UDM
                ndvi_data[sensor]['data'][ndvi_data[sensor]['data']==0] = np.nan

    
    # for reference of where to randomly sample polygons use the planet ROI
    # TODO: just save this as a geojson in the download steps
    
    with rasterio.open(scene_combo['raster_files']['planet']) as src:
        reference_boundary = box(
            minx = src.bounds.left,
            miny = src.bounds.bottom,
            maxx = src.bounds.right,
            maxy = src.bounds.top
            )
        left_bound = src.bounds.left
        right_bound = src.bounds.right
        bottom_bound = src.bounds.bottom
        top_bound    = src.bounds.top
        reference_crs = src.crs

    # Confirm meters since random placements assume this.
    assert reference_crs.linear_units == 'metre', 'reference_crs not in meters'

    for window_size in window_sizes: # meters
        random_points_x = np.random.uniform(low = left_bound, high = right_bound, size=n_random_samples)
        random_points_y = np.random.uniform(low = bottom_bound, high = top_bound, size=n_random_samples)
        random_windows = [box_from_midpoint_and_size(x,y,window_size) for x, y in zip(random_points_x, random_points_y)]
        
        # ensure they are fully inclused within reference
        random_windows = [w for w in random_windows if reference_boundary.contains(w)]
        
        print('processing {} - window_size: {}'.format(scene_combo['ltar_domain'], window_size))
        for w_i, window in enumerate(tqdm(random_windows)):
            pass
            for sensor, sensor_info in ndvi_data.items():
                pass
                if reference_crs != sensor_info['profile']['crs']:
                    window_adjusted = transform_shape_crs(window, reference_crs, sensor_info['profile']['crs'])
                else:
                    window_adjusted = window
                
                data_mask = rasterio.features.geometry_mask(
                    geometries = [window_adjusted],
                    out_shape = sensor_info['data'].shape[1:],
                    transform = sensor_info['profile']['transform'],
                    all_touched=True,
                    invert=True
                    )
                
                # Note window_data gets flattened to a single dimensions, so loses
                # spatial structure. If I end up doing entorpy things which
                # require that I'll have to revisit. Probably using rasterio.features.geometry_window
                window_data = sensor_info['data'][0][data_mask]
                
                # nan values from cloud masks, raster, edge etc.
                # size=0 when window is accidently out outside raster.
                if np.isnan(window_data).any() or window_data.size==0:
                    entropy = np.nan
                    diff_entropy = np.nan
                    mean_ndvi = np.nan
                else:
                    entropy = entropy_calculation(window_data, pixel_size=sensor_pixel_size[sensor])
                    diff_entropy = differential_entropy(window_data)
                    mean_ndvi = window_data.mean()
                
                window_area = window_data.size * (sensor_pixel_size[sensor]**2)
                
                # Collection a list of dictionaries might get a bit memory heavy
                # but we'll see.
                all_results.append(dict(
                    ltar_domain = scene_combo['ltar_domain'],
                    sensor = sensor,
                    window_size = window_size,
                    window_area = window_area,
                    random_window_i = w_i,
                    entropy = entropy,
                    diff_entropy = diff_entropy,
                    mean_ndvi = mean_ndvi,
                    ))
                        
pd.DataFrame(all_results).to_csv(entropy_results_file, index=False)


# Iterate thru the images again and pull actual NDVI values for 
# histogram comparison

n_ndvi_values_for_histograms = 100000

all_ndvi_results = []

for scene_combo in all_scene_combos:
    pass
    # Preload everything into memory
    ndvi_data = {}
    for sensor, raster_file in scene_combo['raster_files'].items():
        ndvi_data[sensor] = []
        with rasterio.open(raster_file) as src:
            data = src.read().flatten()
           
        if sensor=='planet':
            # TODO: Naively set planet nodata until I can do it with the UDM
            data = data[data!=0]
        
        # random sample of ndvi values
        ndvi_data[sensor] = np.random.choice(data, size=n_ndvi_values_for_histograms, replace=False)
    
    ndvi_data = pd.DataFrame(ndvi_data)
    ndvi_data['ltar_domain'] = scene_combo['ltar_domain']
    ndvi_data['scene_id'] = np.nan # TODO
    all_ndvi_results.append(ndvi_data)
        
pd.concat(all_ndvi_results).to_csv(ndvi_sample_file, index=False)

















