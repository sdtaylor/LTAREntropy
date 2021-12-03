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
from shapely.geometry import shape 
import requests
from requests.auth import HTTPBasicAuth
from planet import api
from planet.api import filters


import unpackqa

from tools.io import write_raster
from tools.geotools import get_raster_crop_from_box, transform_4326_shape_to_raster_crs
import planet_tools

import geopandas as gpd

# TODO: more planet processing
# - rename the file similar to other, ie. "jer_planet_ndvi.tif"
# - use UDM to do nodata masking

image_folder = './data/imagery/'

planet_scene_type = 'PSScene4Band'

#------------------------
# Planet API
from planet_api_setup import planet_client, planet_auth

#------------------------
# MS planetary computer STAC clients
from pystac_client import Client
import planetary_computer as pc


ms_l8_id = 'landsat-8-c2-l2'
ms_s2_id = 'sentinel-2-l2a'

ms_catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
ms_l8_catalog = ms_catalog.get_child('landsat-8-c2-l2')
ms_s2_catalog = ms_catalog.get_child('sentinel-2-l2a')
#------------------------


all_scene_combos = pd.read_csv('./data/scene_combinations.csv')

# make  a small test set of scene combos
all_scene_combos = all_scene_combos[(~all_scene_combos.s2_scene_id.isna()) & (~all_scene_combos.l8_scene_id.isna())]
all_scene_combos = all_scene_combos.sample(10, random_state=3).reset_index()

#scene_combo = all_scene_combos[0]
for scene_i, scene_combo in all_scene_combos.iterrows():
    if not scene_combo.scenes_available:
        continue

    print('processing scene {}/{} - {}'.format(scene_i, len(all_scene_combos), scene_combo.ltar_roi_id))
    # domain IDs are like 'wgew-1'
    ltar_domain = scene_combo.ltar_roi_id.split('-')[0]

    # get the planet scene ROI to crop with
    planet_scene_info = planet_client.get_item(planet_scene_type, scene_combo.planet_scene_id).get()
    planet_scene_shape = shape(planet_scene_info['geometry'])
    
    #------------------------------------------
    # L8 NDVI
    
    l8_scene_info = [i.to_dict() for i in ms_catalog.search(ids = [scene_combo.l8_scene_id]).get_items()][0]
    
    l8_bands = dict(SR_B4=None,SR_B5=None,QA_PIXEL=None)
    
    for band in l8_bands.keys():
        band_url = pc.sign(l8_scene_info['assets'][band]['href'])
        
        band_profile, band_data = get_raster_crop_from_box(
            raster_filepath = band_url, 
            box = transform_4326_shape_to_raster_crs(planet_scene_shape, band_url),
            bands='single'
            )
        
        l8_bands[band] = dict(profile=band_profile, data=band_data)
    
    # unpack QA band to relavant flags for masking
    qa_unpacked = unpackqa.unpack_to_array(
        l8_bands['QA_PIXEL']['data'], 
        'LANDSAT_8_C2_L2_QAPixel',
        ['Fill','Dilated_Cloud','Cirrus','Cloud','Cloud_Shadow','Snow','Water'])
    # True where a pixel has any of the above flags
    qa_unpacked = qa_unpacked.sum(axis=-1) > 0
    l8_bands['masked_pixels'] = qa_unpacked
    qa_unpacked = None
    # NDVI and mark masked pixels as na
    b4 = (l8_bands['SR_B4']['data'] * 0.0000275) + -0.2
    b5 = (l8_bands['SR_B5']['data'] * 0.0000275) + -0.2
    l8_ndvi = (b5-b4)/(b5+b4)
    l8_ndvi[l8_bands['masked_pixels']] = np.nan
    
    l8_ndvi_profile = l8_bands['SR_B4']['profile']
    l8_ndvi_profile['dtype'] = rasterio.float32
    
    raster_filename =  '{}_{}_l8_ndvi.tif'.format(scene_combo.ltar_roi_id, scene_combo.time_period)
    write_raster(filepath = path.join(image_folder, raster_filename),
                 raster_profile = l8_ndvi_profile,
                 raster_data = l8_ndvi,
                 bands = 'single'
                 )
    
    #------------------------------------------
    # Sentinal 2 NDVI
    s2_scene_info = [i.to_dict() for i in ms_catalog.search(ids = [scene_combo.s2_scene_id]).get_items()][0]
    
    s2_bands = dict(B08=None,B04=None,SCL=None)
    
    for band in s2_bands.keys():
        band_url = pc.sign(s2_scene_info['assets'][band]['href'])
        
        band_profile, band_data = get_raster_crop_from_box(
            raster_filepath = band_url, 
            box = transform_4326_shape_to_raster_crs(planet_scene_shape, band_url),
            bands='single'
            )
        
        s2_bands[band] = dict(profile=band_profile, data=band_data)
        
    # S2 scene classifiction map, where 4 & 5 are "vegetated" & "non-vegetated", and everthting
    # else is clouds, cloud shadow, etc. 6 is water
    s2_mask = np.logical_or(s2_bands['SCL']['data'] == 4, s2_bands['SCL']['data'] == 5)
    # invert and change to int so 1 is masked pixels
    s2_mask = (~s2_mask).astype(np.uint8)
    s2_mask_placeholder = s2_bands['B08']['data'].copy()
    
    # s2 SCL is 20m resolution, so resample using nearest neighber to 10m 
    # to match ndvi
    s2_mask_resampled, s2_mask_resampled_profile = rasterio.warp.reproject(
        source = s2_mask,
        destination = s2_mask_placeholder,
        src_transform = s2_bands['SCL']['profile']['transform'],
        dst_transform = s2_bands['B08']['profile']['transform'],
        src_crs = s2_bands['SCL']['profile']['crs'],
        dst_crs = s2_bands['B08']['profile']['crs'],
        )
    
    
    # NDVI and marked 
    b8 = s2_bands['B08']['data'] * 0.0001
    b4 = s2_bands['B04']['data'] * 0.0001
    s2_ndvi = (b8 - b4)/(b8 + b4)
    s2_ndvi[s2_mask_resampled == 1] = np.nan
    
    s2_ndvi_profile = s2_bands['B08']['profile']
    s2_ndvi_profile['dtype'] = rasterio.float32
    raster_filename = '{}_{}_s2_ndvi.tif'.format(scene_combo.ltar_roi_id, scene_combo.time_period)
    write_raster(filepath = path.join(image_folder, raster_filename),
                 raster_profile = s2_ndvi_profile,
                 raster_data = s2_ndvi,
                 bands = 'single'
                 )
    