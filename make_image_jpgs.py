import glob
from os import path

import numpy as np
import rasterio

from tqdm import tqdm

from tools.io import write_jpg


""" 
Make jpgs of all s2/l8/planet images for doing quick visual diagnostics.
This will be black/white representing NDVI
"""

jpg_folder = './data/imagery_thumbnails/'
image_folder = './data/imagery/'


all_tif_files = glob.glob(image_folder + '*.tif')


for f in tqdm(all_tif_files):
    pass

    with rasterio.open(f) as src:
        tif_data = src.read(1)
        tif_profile = src.profile
        
    new_jpg_filename = path.basename(f).split('.')[0] + '.jpg'
    new_jpg_filepath = path.join(jpg_folder, new_jpg_filename)
    
    annotation = path.basename(f).split('.')[0]
    
    write_jpg(filepath = new_jpg_filepath, 
              raster_data = tif_data, 
              annotation=annotation, 
              bands='single', 
              data_format='channels_first', 
              show_image=False
        )