import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
from requests.auth import HTTPBasicAuth

from shapely.geometry import Polygon, MultiPolygon, shape
import geopandas as gpd


_ORDERS_URL = 'https://api.planet.com/compute/ops/orders/v2'
_HEADERS = {'content-type': 'application/json'}

# define helpful functions for submitting, polling, and downloading an order
def place_order(request, auth):
    response = requests.post(_ORDERS_URL, data=json.dumps(request), auth=auth, headers=_HEADERS)
    print(response)
    
    if not response.ok:
        raise Exception(response.content)

    order_id = response.json()['id']
    print(order_id)
    order_url = _ORDERS_URL + '/' + order_id
    return order_url

def wait_on_order(order_url, auth, timeout=600, check=10):
    """ 
    block until the planet order completes (failure or success), or
    until the timeout (seconds)
    """
    
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        success_states = ['success', 'partial']
        if state == 'failed':
            raise Exception(response)
        elif state in success_states:
            break
        
        time.sleep(check)
        
def download_order(order_url, auth, dest_dir, overwrite=False):
    r = requests.get(order_url, auth=auth)
    print(r)

    response = r.json()
    results = response['_links']['results']
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    results_paths = [pathlib.Path(os.path.join(dest_dir, n)) for n in results_names]
    print('{} items to download'.format(len(results_urls)))
    
    for url, name, path in zip(results_urls, results_names, results_paths):
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path, name))
            
    return dict(zip(results_names, results_paths))

def item_list_to_footprints(items):
    """
    From a list of search query items, create a GeoDataFrame of the item
    outlines along with various attributes.
    """
    attribute_names = ['satellite_id','acquired','clear_percent','cloud_percent']
    
    attributes = []
    geoms = []
    for i in items:
        attributes.append({a:i['properties'][a] for a in attribute_names})
        geoms.append(shape(i['geometry']))
    
    geodf = gpd.GeoDataFrame(attributes, geometry=gpd.GeoSeries(geoms)).set_crs(4326)
    geodf['date'] = gpd.pd.DatetimeIndex(geodf.acquired).strftime('%Y-%m-%d')
    geodf['time'] = gpd.pd.DatetimeIndex(geodf.acquired).strftime('%H%M')
    
    return geodf
