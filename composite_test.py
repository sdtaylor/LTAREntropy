import datetime
import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
from requests.auth import HTTPBasicAuth

from planet import api
from planet.api import filters
import planet_tools

import geopandas as gpd

api_key = 'PASTE KEY HERE BUT DO NOT COMMIT'
PLANET_API_KEY = os.environ.get('PL_API_KEY', api_key)

client = api.ClientV1(api_key=PLANET_API_KEY)


# set up requests to work with api
auth = HTTPBasicAuth(PLANET_API_KEY, '')


#----------------------------
# Get tile ID's to composite

date_range_start =  datetime.datetime(year=2020,month=5,day=15)
date_range_end =  datetime.datetime(year=2020,month=5,day=17)

jornada_roi = {
        "type": "Polygon",
        "coordinates": [
          [[-106.875,   32.514973],
            [-106.706085,32.514973],
            [-106.706085,32.640531],
            [-106.875,   32.640531],
            [-106.875,   32.514973]]
        ]}

query = filters.and_filter(
    filters.geom_filter(jornada_roi),
    filters.range_filter('clear_percent', gte=90),
    filters.date_range('acquired', gte=date_range_start),
    filters.date_range('acquired', lte=date_range_end)
)

request = filters.build_search_request(query, ['PSScene4Band'])

# search the data api
result = client.quick_search(request)
items = list(result.items_iter(limit=500))

item_outlines = planet_tools.item_list_to_footprints(items)
item_outlines.to_file('./item_outline_test.geojson', driver='GeoJSON')

products_order_info = [
    {'item_ids': [i['id'] for i in items],
     'item_type': 'PSScene4Band',
     'product_bundle' : 'analytic'
     }
    ]
#---------------------------
# Order tools
clip = {"clip": {"aoi": jornada_roi}}
bandmath = {"bandmath": {
    "pixel_type": "32R",
    "b1": "(b4 - b3) / (b4 + b3)"
  }
}
composite = {"composite":{}}


order_request = {
    'name' : 'jornada test',
    'products' : products_order_info,
    'tools' : [composite, clip, bandmath]
    }

order_url = planet_tools.place_order(order_request, auth)
planet_tools.poll_for_success(order_url, auth)

downloaded_files = planet_tools.download_order(order_url, auth)








