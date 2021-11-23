import rasterio
from rasterio import windows
from skimage.transform import resize

import numpy as np
import geopandas as gpd

from shapely.geometry import box, Polygon

from math import cos, radians
import datetime
import json
from os import path, mkdir

import pyproj
from shapely.ops import transform as shapely_transform

def transform_4326_shape_to_raster_crs(polygon, raster_path):
    """ 
    Given any shapely shape in the lat/lon crs 4326, transfrom to the
    crs of the raster at raster_path.
    """
    wgs84 = pyproj.CRS('EPSG:4326')
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    project = pyproj.Transformer.from_crs(wgs84, raster_crs, always_xy=True).transform
    return shapely_transform(project, polygon)

def transform_shape_crs(shape, src_crs, dst_crs):
    """Transform a shapely shape to an arbitrary crs"""
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shapely_transform(project, shape)

def get_raster_crop_from_box(raster_filepath, 
                             box,
                             bands='rgb'):
    """
    Get a cropped raster as a numpy array, and a rasterio.profile with
    the adjusted transform information.
    
    It's assumed that box is already in the crs coordinates of the raster.

    Parameters
    ----------
    raster_filepath : str
        Path to raster file. The full raster will not be loaded into memory.
    box : shapely.geometry.Polygon
        The box to crop with. Can actually be any polygon, but only the 
        min/max bounds will be used.
    bands : str, optional
        Either 'rgb' or 'single' for true color or single band tifs, respectively. 
        The default is 'rgb'.
        
    Returns
    -------
    cropped_profile : dict
        rasterio style profile
    cropped_raster_data : array
        numpy array of raster data with shape (bands, height, width)

    """
    if bands == 'rgb':
        bands = [1,2,3]
    elif bands == 'single':
        bands = [1]
    # also accept a multi-band list like [1,2,3,4,5,6]
    elif isinstance(bands, list) and all(isinstance(b, int) for b in bands):
        pass
    else:
        raise ValueError('bands should be "rgb","single" or list of band numbers')
        
    with rasterio.open(raster_filepath) as src:
            src_raster_profile = src.profile
                        
            # The rasterio window defines the rows,cols of the cropping box
            minx, miny, maxx, maxy = box.bounds
            win = windows.from_bounds(left   = minx, 
                                      bottom = miny, 
                                      right  = maxx,
                                      top    = maxy,
                                      transform = src_raster_profile['transform'])
            
            # This has the shape (bands, height, width) even with just 1 band
            cropped_raster_data = src.read(bands, window=win)
            window_transform =  windows.transform(win, src_raster_profile['transform'])
            
            cropped_profile = src_raster_profile.copy()
            cropped_profile.update({'height'    : cropped_raster_data.shape[1],
                                    'width'     : cropped_raster_data.shape[2],
                                    'count'     : len(bands),
                                    'transform' : window_transform})
    
    # These settings don't translate well from large to small images, causing errors
    # without them gdal will just set a default value.
    for attr_to_drop in ['blockysize','blockxsize']:
        if attr_to_drop in cropped_profile:
            _ = cropped_profile.pop(attr_to_drop)
    
    return cropped_profile, cropped_raster_data



