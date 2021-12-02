import rasterio
import matplotlib.pyplot as plt


def write_jpg(filepath, raster_data, annotation=None, 
              bands='rgb', data_format='channels_first', 
              show_image=False):
    """
    Write an image with a single text box at the top
    """
    if bands == 'rgb':
        if data_format=='channels_first':
            assert raster_data.shape[0] <= 4, 'data_format set to channels_first but axis 0 has shape > 4'
            raster_data = np.moveaxis(raster_data,0,2)
        elif data_format=='channels_last':
            assert raster_data.shape[2] <= 4, 'data_format set to channels_last but axis 2 has shape > 4'
        else:
            raise ValueError('data_format must be channels_first or channels_last')
    elif bands == 'single':
        pass
    else:
        raise ValueError('bands must be single or rgb')
    
    # Diable images poping up in ipython
    if show_image:
        plt.ion()
    else:
        plt.ioff()
    
    fig, ax = plt.subplots()
    ax.imshow(raster_data)
    
    width, height = raster_data.shape[1], raster_data.shape[0]
    
    text_xy = (int(width/2), int(height*0.05))
    if annotation:
        assert isinstance(annotation, str), 'annotation must be a string'
        ax.annotate(annotation, 
                    xy=text_xy, 
                    horizontalalignment='center', 
                    bbox=dict(facecolor='white'))
    
    plt.axis('off')
    
    plt.savefig(filepath,bbox_inches='tight',pad_inches=0)

def write_raster(filepath, raster_profile, raster_data, bands='rgb'):
    """
    Write a raster with profile and data output from get_aws_crop_from_point()
    Parameters
    ----------
    filepath : str
        Full path and filename to write to.
    raster_profile : dict
        rasterio style profile.
    raster_data : array
        numpy array
    bands : str or list, optional
        Either 'rgb' or 'single' for true color or single band tifs, respectively.
        Or a list of band numbers like [1,2,3,4,5,6]
        The default is 'rgb'.
    Returns
    -------
    None.
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

    with rasterio.open(filepath, 'w', **raster_profile) as dst:
        dst.write(raster_data, indexes=bands)
