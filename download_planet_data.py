import datetime
import json
import os
from os import path
import time

import glob

import pandas as pd
import numpy as np
import rasterio
import rasterio.warp
from rasterio.plot import show

from planet.api import filters


import unpackqa

from tools.io import write_raster
from tools.geotools import get_raster_crop_from_box, transform_4326_shape_to_raster_crs
import planet_tools

import geopandas as gpd

#----------------------------

image_folder = './data/imagery/'

planet_temp_folder = './data/imagery/planet_download/'

planet_scene_type = 'PSScene4Band'

#------------------------
# Planet API
from planet_api_setup import planet_client, planet_auth


#------------------------

all_scene_combos = pd.read_csv('./data/scene_combinations.csv')

# make  a small test set of scene combos
all_scene_combos = all_scene_combos[(~all_scene_combos.s2_scene_id.isna()) & (~all_scene_combos.l8_scene_id.isna())]
all_scene_combos = all_scene_combos.sample(10, random_state=3).reset_index()

    
#-------------------------------------
# For the planet scenes, put in a single order for all of them to download
# at once. Then iterate thru and put them next to sentinal2/landsat8

#------------------------------------------
# Planet NDVI

def submit_and_download_order(scene_list, dest_dir):
    products_order_info = [
    {'item_ids': scene_list,
     'item_type': 'PSScene4Band',
     'product_bundle' : 'analytic_sr_udm2'
     }
    ]
    #---------------------------
    # Order tools
    bandmath = {"bandmath": {
        "pixel_type": "32R",             # the resulting dtype, 32R = 32bit floating point
        "b1": "(b4 - b3) / (b4 + b3)",   # NDVI
        "b2":  "2.5 * ((b4 - b3) / (1 + b4 + (2.4*b3)))",  # EVI2 (2 band evi)
        "b3": "b3",  # red band
        "b4": "b4"   # nir band
      }
    }
    
    order_request = {
        'name' : 'LTAREntropy test dec2',
        'products' : products_order_info,
        'tools' : [bandmath]
        }
    
    order_url = planet_tools.place_order(order_request, planet_auth)
    planet_tools.wait_on_order(order_url, planet_auth)
    
    # wait on order processing
    
    downloaded_files = planet_tools.download_order(order_url, planet_auth, dest_dir=dest_dir)

    # wait on download
    
    # a list of file downloads
    return downloaded_files

def process_planet_imagery(source_dir, dest_dir):
    """ 
    Make single band NDVI/EVI files with bad pixels masked out using udm2
    
    Expected that source_dir will be full of downloaded planet from the  submit and 
    download order function.
    Will produce ndvi files inside dest_dir like: gacp-3_2020-07-20_planet_ndvi.tif
    """
    search_str = source_dir + '/**/*.tif'
    all_downloaded_files = glob.glob(search_str, recursive=True)
    
    for scene_i, scene_combo in all_scene_combos.iterrows():
        if not scene_combo.scenes_available:
            continue
        
        scene_planet_files = [f for f in all_downloaded_files if scene_combo.planet_scene_id in f]
        primary_file = [f for f in scene_planet_files if 'bandmath.tif' in f]
        assert len(primary_file) == 1
        primary_file = primary_file[0]
        udm2_file = [f for f in scene_planet_files if 'udm2.tif' in f]
        assert len(udm2_file) == 1
        udm2_file = udm2_file[0]
    
        
        with rasterio.open(primary_file) as primary_src, rasterio.open(udm2_file) as udm2_src:
            planet_ndvi = primary_src.read(1) # band 1 is ndvi in the bandmath done above
            planet_ndvi_profile = primary_src.profile
            planet_clear = udm2_src.read(1) # band 1 of udm2 is the  "clear map" where 1=clear and 0=not clear, https://developers.planet.com/docs/data/udm-2/
    
        planet_ndvi[planet_clear==0] = np.nan
    
        raster_filename = '{}_{}_planet_ndvi.tif'.format(scene_combo.ltar_roi_id, scene_combo.time_period)
        planet_ndvi_profile.update(count=1)
        write_raster(filepath = path.join(image_folder, raster_filename),
                 raster_profile = planet_ndvi_profile,
                 raster_data = np.expand_dims(planet_ndvi,0),
                 bands = 'single'
                 )
