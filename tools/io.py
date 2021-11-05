import rasterio


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
