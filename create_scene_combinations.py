import datetime
import json
import os
import pathlib
import time

import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from shapely.geometry import shape 
import requests
from requests.auth import HTTPBasicAuth
from planet import api
from planet.api import filters
import planet_tools

from CompositePrep import planet_scene_trimming

import geopandas as gpd

"""
Given some  target ROI's and focal dates, find planet scenes within those bounds.
Then find matching Landsat 8 and Sentinal 2 scenes with minimal clouds on the MS planetary computer through stac search.
Create a csv of results and geojson of resulting image bounds.
"""

from planet_api_setup import planet_client, planet_auth

#------------------------
# MS planetary computer STAC clients
from pystac_client import Client

ms_l8_id = 'landsat-8-c2-l2'
ms_s2_id = 'sentinel-2-l2a'

ms_catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
ms_l8_catalog = ms_catalog.get_child('landsat-8-c2-l2')
ms_s2_catalog = ms_catalog.get_child('sentinel-2-l2a')

#-------------------------
# Setup search space across space and  time

ltar_rois = gpd.read_file('./data/ltar_region_rois.geojson')
#ltar_rois = ltar_rois[ltar_rois.roi_id=='umrb-1']

all_rois = json.loads(ltar_rois.to_json())

focal_dates = [
    datetime.datetime(year=2020,month=5,day=20),
    datetime.datetime(year=2020,month=7,day=20),
    datetime.datetime(year=2020,month=9,day=20)
    ]

planet_search_window = 6 # +- this many days
l8_search_window = 9
s2_search_window = 8

max_planet_scenes_per_roi = 30

#---------------------------


def get_search_window(focal_date, window_size, api='planet'):
    date_range_start = focal_date - datetime.timedelta(days=window_size)
    date_range_end   = focal_date + datetime.timedelta(days=window_size)
    
    if api == 'planet':
        return date_range_start, date_range_end
    elif api == 'stac':
        # STAC date search str looks like '2019-01-01/2019-01-31'
        return '{}/{}'.format(date_range_start.strftime('%Y-%m-%d'), date_range_end.strftime('%Y-%m-%d'))
    else:
        raise ValueError(f'unknown api type: {api}')




def find_adquate_scene(stack_items, roi, max_cloud_coverage=10, min_roi_coverage=0.95):
    """ 
    Return the first scene from stack items which meets the min. requirements
    of max cloud coverage and minimum area coverage of the ROI.
    
    Returns None if none of them meet the requirements.
    """
    roi_shape=shape(roi)
    
    # sort by cloud cover to return the one w/ the lowest first. 
    stack_items = sorted(stack_items, key= lambda x: x.properties['eo:cloud_cover'])
    
    for item in stack_items:
        pass
        # adequate clouds?
        if item.properties['eo:cloud_cover'] > max_cloud_coverage:
            continue
        # adequate coverage?
        scene_outline = shape(item.geometry)
        scene_coverage = roi_shape.intersection(scene_outline).area / roi_shape.area
        if scene_coverage < min_roi_coverage:
            continue
        
        return item

    return None

def stack_item_to_metadata(item):
    outline = dict(
        item_found = item != None
        )

def stack_list_roi_coverage(stack_items, roi):
    """ 
    Given a list of STAC items and an roi geom dictonary, calculate % coverage
    of each stack item with the roi
    """
    roi_shape = shape(roi)
    
    to_return = []
    for item in stack_items:
        item_shape = shape(item.geometry)
        percent_coverage = roi_shape.intersection(item_shape).area/roi_shape.area
        to_return.append(percent_coverage)
    
    return to_return
        
item_search_results = []
search_result_shapes = {
    'geometry' : [],
    'type'     : [],
    'id'       : [],
    'time_period' : []
    }

n_rois = len(all_rois['features'])

for roi_i, roi_feature in enumerate(all_rois['features']):
    roi = roi_feature['geometry']
    roi_id = roi_feature['properties']['roi_id']
    roi_shape = shape(roi)
    
    ltar_domain = roi_id.split('-')[0]
    
    print(f'processing roi {roi_i}/{n_rois} {roi_id}')
    # jornada_roi = {
    #         "type": "Polygon",
    #         "coordinates": [
    #           [[-106.875,   32.514973],
    #            [-106.706085,32.514973],
    #            [-106.706085,32.640531],
    #            [-106.875,   32.640531],
    #            [-106.875,   32.514973]]
    #         ]}
    
    for focal_date in focal_dates:
    
        planet_search_date_start, planet_search_date_end = get_search_window(
            focal_date = focal_date,
            window_size = planet_search_window,
            api = 'planet'
            )
        
        query = filters.and_filter(
            filters.geom_filter(roi),
            filters.range_filter('clear_percent', gte=90),
            filters.date_range('acquired', gte=planet_search_date_start),
            filters.date_range('acquired', lte=planet_search_date_end)
        )
        
        request = filters.build_search_request(query, ['PSScene4Band'])
        
        # search the data api
        result = planet_client.quick_search(request)
        items = list(result.items_iter(limit=5000))
        
        # Limit to 1 instrument type. This can potentially be ajusted if clouds
        # end up being too much
        items = [i for i in items if i['properties']['instrument'] == 'PS2.SD']
        
        # Sort by cloudiness
        items = sorted(items, key = lambda i: -i['properties']['clear_percent'])
        
        # how many candidate scenes have been evaled for this ROI?
        roi_planet_scene_count = 0
        
        for planet_scene in items:
            roi_planet_scene_count += 1
            if roi_planet_scene_count > max_planet_scenes_per_roi:
                break   
            
            # Given a potential planet scene to use, see if there is a corresponding
            # L8 and S2 image within the desired date range. 
            planet_scene_roi = planet_scene['geometry']
            planet_scene_roi_shape = shape(planet_scene_roi)
            # TODO: check urban/water coverage
            search_result_shapes['geometry'].append(planet_scene_roi_shape)
            search_result_shapes['type'].append('planet')
            search_result_shapes['id'].append(planet_scene['id'])
            search_result_shapes['time_period'].append(focal_date)
            
            
            # STAC date search str looks like '2019-01-01/2019-01-31'
            l8_date_str =  get_search_window(
                    focal_date = focal_date,
                    window_size = l8_search_window,
                    api = 'stac'
                )
            
            # landsat 8 search
            l8_scenes = ms_catalog.search(collections = [ms_l8_id],
                                          intersects  = planet_scene_roi,
                                          datetime    = l8_date_str)
            l8_scenes = [i for i in l8_scenes.get_items()]
            l8_scene = find_adquate_scene(l8_scenes, roi=planet_scene_roi)
           
            l8_scene_info = dict(
                l8_scene_found = False,
                l8_scene_id = None,
                l8_scene_clouds = None,
                l8_scene_roi_coverage = None
                )
            if l8_scene:
                l8_scene_info['l8_scene_found'] = True
                l8_scene_info['l8_scene_id'] = l8_scene.id
                l8_scene_info['l8_scene_clouds'] = round(l8_scene.properties['eo:cloud_cover'],2)
                l8_scene_info['l8_scene_roi_coverage'] = round(planet_scene_roi_shape.intersection(shape(l8_scene.geometry)).area / planet_scene_roi_shape.area,2)
    
    
                if l8_scene.id not in search_result_shapes['id']:
                    search_result_shapes['geometry'].append(shape(l8_scene.geometry))
                    search_result_shapes['type'].append('l8')
                    search_result_shapes['id'].append(l8_scene.id)
                    search_result_shapes['time_period'].append(focal_date)
             
           
            # sentinal 2 search
            s2_date_str =  get_search_window(
                    focal_date = focal_date,
                    window_size = s2_search_window,
                    api = 'stac'
                )
            s2_scenes = ms_catalog.search(collections = [ms_s2_id],
                                          intersects  = planet_scene_roi,
                                          datetime    = s2_date_str)
            s2_scenes = [i for i in s2_scenes.get_items()]
            
            s2_scene = find_adquate_scene(s2_scenes, roi=planet_scene_roi)
            
            s2_scene_info = dict(
                s2_scene_found = False,
                s2_scene_id = None,
                s2_scene_clouds = None,
                s2_scene_roi_coverage = None
                )
            if s2_scene:
                s2_scene_info['s2_scene_found'] = True
                s2_scene_info['s2_scene_id'] = s2_scene.id
                s2_scene_info['s2_scene_clouds'] = round(s2_scene.properties['eo:cloud_cover'],2)
                s2_scene_info['s2_scene_roi_coverage'] = round(planet_scene_roi_shape.intersection(shape(s2_scene.geometry)).area / planet_scene_roi_shape.area,2)
    
                if s2_scene.id not in search_result_shapes['id']:
                    search_result_shapes['geometry'].append(shape(s2_scene.geometry))
                    search_result_shapes['type'].append('s2')
                    search_result_shapes['id'].append(s2_scene.id)
                    search_result_shapes['time_period'].append(focal_date)
            
            planet_scene_search_result = dict(
                ltar_domain = ltar_domain,
                ltar_roi_id = roi_id,
                time_period = str(focal_date.date()),
                planet_scene_id = planet_scene['id'],
                scene_clear_percent = planet_scene['properties']['clear_percent'],
                **l8_scene_info,
                **s2_scene_info
                )
            
            item_search_results.append(planet_scene_search_result)
        
pd.DataFrame(item_search_results).to_csv('./roi_search_results.csv',index=False)

search_result_shapes = pd.DataFrame(search_result_shapes)

for t in focal_dates:
    temp_df = search_result_shapes[search_result_shapes.time_period == t]
    shapefile_name = './data/roi_search_shapes_{}.geojson'.format(t.date())
    gpd.GeoDataFrame(temp_df).set_crs(4326).to_file(shapefile_name, driver='GeoJSON')

