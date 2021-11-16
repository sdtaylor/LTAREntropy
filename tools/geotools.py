import numpy as np


def resample_planet_imagery(original_img, dest_res, src_res=3):
    """
    Upscale the ~3m planet imagery to a coarser resolution.
    
    Uses fixed windows, not sliding windows, via the view_as_blocks method
    in skimage
    The number of pixels is *retained*

    Parameters
    ----------
    original_img : TYPE
        DESCRIPTION.
    res : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert isinstance(original_img, np.ndarray), 'original_img must be np array'
    assert len(original_img.shape) == 2, 'original_img must be 2D array'
    window_size = np.floor(dest_res/src_res).astype(int)
    
    # crop the img to fit the new resolution exactly.
    # a few meters lost around the edges is not big deal
    n_windows_y = np.floor(original_img.shape[0] / window_size).astype(int)
    n_windows_x = np.floor(original_img.shape[1] / window_size).astype(int)
    
    original_img = original_img[:n_windows_y * window_size, :n_windows_x*window_size]
    
    windows = view_as_blocks(original_img, block_shape=(window_size,window_size))
    
    windows2 = np.zeros_like(windows, dtype=windows.dtype)
    # TODO: explain this part. This makes the tiled window mean while 
    # retaining the original resolution. In other words, a 3m resolution image
    # upscaled to 30m keeps the same number of pixels, but everything in 
    # 30m chunks has the same mean value.
    windows2.T[:] = windows.mean(axis=(2,3)).T
    
    output_size = (n_windows_y * window_size, n_windows_x * window_size)
    # See here on reversing view_as_blocks() https://stackoverflow.com/a/59938823
    return windows2.transpose(0,2,1,3).reshape(output_size)
