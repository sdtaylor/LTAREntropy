import datetime
import json
import os
from os import path
import time
import glob

import pandas as pd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box

from tools.geotools import transform_shape_crs

from tqdm import tqdm


n_random_samples=1000
window_sizes = [500] # in meters
entropy_results_file = './data/entropy_results_NDVI.csv'
abs_value_sample_file = './data/scene_absolute_value_samples.csv'

sensor_pixel_size = dict(
    planet=3,
    l8=30,
    s2=10
    )


image_folder = './data/imagery/'

#----------------
# From all tif files create the scene combos, where for each roi_id  and time_period
# there is a single planet, s2, or l8 scene. Where l8 OR s2 are potentially missing,
# but at least one is always paired with a planet scene.
# For loops below are setup such that all scenes, with multiple indices, are loaded
# and processed at once for any give roi and time period. This ensures that, for
# a single  randomly placed window, the same area/time is sampled for all sensors.

all_tif_files = glob.glob(image_folder + '*.tif')

def extract_file_info(filepath):
    """  
    from filenames like: ecb-1_2020-09-20_planet_ndvi.tif
    """
    filename_split = path.basename(filepath).split('.')[0].split('_')
    return dict(
        ltar_domain = filename_split[0].split('-')[0],
        roi_id = filename_split[0],
        time_period = filename_split[1],
        sensor = filename_split[2],
        index = filename_split[3],
        image_filepath = filepath,
        )

all_scenes = pd.DataFrame([extract_file_info(f) for f in all_tif_files])

all_scenes['scene_combo_id'] = all_scenes.roi_id + '_' + all_scenes.time_period

# a scene combination is all scenes for a single roi/time_period
all_scene_combos = all_scenes[['ltar_domain','roi_id','time_period','scene_combo_id']].drop_duplicates().reset_index()


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

for scene_combo_i, scene_combo in all_scene_combos.iterrows():
    pass

    scene_combo_files = all_scenes[all_scenes.scene_combo_id==scene_combo.scene_combo_id].image_filepath.to_list()

    # Preload everything into memory
    ndvi_data = {}
    for scene_file in scene_combo_files:
        ndvi_data[scene_file] = {}
        #sensor_raster_file = scene_combo[sensor]
        
        # Not every scene combo has all 3 sensors
        # Don't actually need the  image_available flag after redoing this a bit
        if not isinstance(scene_file, str):
            ndvi_data[scene_file]['data'] = None
            ndvi_data[scene_file]['profile'] = None
            ndvi_data[scene_file]['image_available'] = False
        else:
            with rasterio.open(scene_file) as src:
                ndvi_data[scene_file]['data'] = src.read()
                ndvi_data[scene_file]['profile'] = src.profile
                ndvi_data[scene_file]['image_available'] = True
            
    
    # for reference of where to randomly sample polygons, use any available planet ROI
    # for this ROI
    ref_file = [ f for f in scene_combo_files if 'planet' in f]
    if len(ref_file) > 0:
        ref_file = ref_file[0]
    else:
        continue
    
    with rasterio.open(ref_file) as src:
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
        
        print('processing combo {}/{} - {} - {} - window_size: {}'.format(
            scene_combo_i+1, all_scene_combos.shape[0],
            scene_combo['roi_id'],
            scene_combo['time_period'], 
            window_size))
        for w_i, window in enumerate(tqdm(random_windows)):
            pass
            for scene_file, scene_data in ndvi_data.items():
                pass
                
                scene_info = extract_file_info(scene_file)
            
                if not scene_data['image_available']:
                    continue

                if reference_crs != scene_data['profile']['crs']:
                    window_adjusted = transform_shape_crs(window, reference_crs, scene_data['profile']['crs'])
                else:
                    window_adjusted = window
                
                data_mask = rasterio.features.geometry_mask(
                    geometries = [window_adjusted],
                    out_shape = scene_data['data'].shape[1:],
                    transform = scene_data['profile']['transform'],
                    all_touched=True,
                    invert=True
                    )
                
                # Note window_data gets flattened to a single dimensions, so loses
                # spatial structure. If I end up doing entorpy things which
                # require that I'll have to revisit. Probably using rasterio.features.geometry_window
                window_data = scene_data['data'][0][data_mask]
                
                sensor = scene_info['sensor']
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
                    roi_id = scene_combo['roi_id'],
                    time_period = scene_combo['time_period'],
                    sensor = sensor,
                    index = scene_info['index'],
                    window_size = window_size,
                    window_area = window_area,
                    random_window_i = w_i,
                    entropy = entropy,
                    diff_entropy = diff_entropy,
                    mean_ndvi = mean_ndvi,
                    ))
                        
pd.DataFrame(all_results).to_csv(entropy_results_file, index=False)


# Iterate thru the images again and pull actual NDVI/EVI values for 
# histogram comparison

n_values_for_histograms = 10000

all_abs_value_results = []

for tif_file in tqdm(all_tif_files):

    scene_info = extract_file_info(tif_file)

    with rasterio.open(tif_file) as src:
            data = src.read().flatten()

    data = data[~np.isnan(data)]
    data = np.random.choice(data, size=n_values_for_histograms, replace=False)

    scene_data = pd.DataFrame({'value':data})
    scene_data['ltar_domain'] = scene_info['ltar_domain']
    scene_data['roi_id'] = scene_info['roi_id']
    scene_data['sensor'] = scene_info['sensor']
    scene_data['index'] = scene_info['index']


    all_abs_value_results.append(scene_data)
        
pd.concat(all_abs_value_results).to_csv(abs_value_sample_file, index=False)

















