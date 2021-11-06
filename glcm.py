import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks, view_as_windows
from skimage.filters.rank import entropy
from skimage.measure import shannon_entropy
from skimage.morphology import square
from skimage.feature.texture import greycomatrix, greycoprops

from scipy import ndimage


def fast_glcm(img, distances=[1], angles=[0], levels=8, kernel_size=5):
    # https://github.com/tzm030329/GLCM/blob/master/fast_glcm.py
    assert img.dtype == np.uint8
    # assert kernel_size is not even
    h,w = img.shape
    n_distances = len(distances)
    n_angles    = len(angles)

    # The offset matrix allows for quick comparison. For example, with 
    # distance=1, angle=0 (compare pixel to the right), the offset matrix
    # has the original img shifted 1 place to the left. This is repeated
    # for all distance/angles so all ij comparisons can happen at once.
    #
    # The kernel is used below in to calculate the moving window sums. The shift
    # with the constant value=0 means the appropriate edge within the window
    # has weight=0, and so is not included in the moving window sum. 
    # For example, with distance=1,angle=0, the right most pixels
    # do not technically have a pixel to compare to, even though within the glcm 
    # there are comparisons made, so their kernel weight is set to 0.
    #
    # see here if ndimage.shift becomes a bottleneck https://stackoverflow.com/a/62841583
    img_offset_matrix = np.zeros((n_distances,n_angles) + img.shape, dtype=img.dtype)
    kernel = np.ones((n_distances,n_angles) + (kernel_size, kernel_size))
    for d_i, d in enumerate(distances):
        for a_i, a in enumerate(angles):
            offset_row = (np.sin(a) * d).round(0).astype(int)
            offset_col = (np.cos(a) * d).round(0).astype(int)
            img_offset_matrix[d_i,a_i] = ndimage.shift(img, (offset_row, -offset_col), mode='nearest')
            
            kernel[d_i, a_i] = ndimage.shift(kernel[d_i,a_i],(offset_row, -offset_col), mode='constant', cval=0)

    # Make masks to mark every instance where i==j.
    glcm = np.zeros((levels, levels, n_distances, n_angles, h, w), dtype=np.uint32)
    for i in range(levels):
        for j in range(levels):
            mask = ((img==i) & (img_offset_matrix==j))
            glcm[i,j, mask] = 1
            
    # Moving window sums of the glcm
    for i in range(levels):
        for j in range(levels):
            for d_i, d in enumerate(distances):
                for a_i, a in enumerate(angles):
                    # correlate is the moving window sum, based on kernel weights
                    glcm[i,j, d_i, a_i] = ndimage.correlate(glcm[i,j, d_i, a_i], kernel[d_i,a_i], mode='nearest')

    #TODO: if normed, do normaliziation.
    #glcm = glcm.astype(np.float32) / glcm.sum((0,1))
    #glcm = glcm.astype(np.float32)
        
    # return minus the padding from the window values. While the ndiimage.correlate step
    # above calculated edge P_ij values, the canonical way to do glcms is to not calculate
    # P_ij values for the edges, and instead interpolate the final stat (eg. entropy, contrast, etc)
    # to the edges to match the original image shape.
    edge_padding = int(kernel_size/2)
    
    return glcm[:,:,:,:,edge_padding:-edge_padding,edge_padding:-edge_padding]

def fast_glcm_contrast(img, nbit=8, kernel_size=3):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img=img, nbit=nbit, kernel_size=kernel_size)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2

    return cont


def glcm_skimage(img, distances=[1], angles=[0], levels=8, kernel_size=3, normed=False):
    """ 
    A moving window glcm where it is calculated for each pixel.
    """
    n_distances = len(distances)
    n_angles    = len(angles)
    
    # skimage glcm does angles in a clockwise fashion, transpose them to be counterclockwise
    angles = [a*-1 for a in angles]
    
    img_windows = view_as_windows(img, window_shape=kernel_size, step=1)
    glcm = np.zeros((levels,levels,n_distances,n_angles)+img_windows.shape[0:2])
    for i in range(img_windows.shape[0]):
        for ii in range(img_windows.shape[1]):
            glcm[:,:,:,:,i,ii] = greycomatrix(img_windows[i,ii], distances, angles, levels, symmetric=False, normed=normed)

    return glcm


def glcm_skimage_stats(img, stats, distances=[1], angles=[0], levels=8, kernel_size=3):
    valid_stats = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
    assert all([s in valid_stats for s in stats]), 'invalid stats'
    
    n_stats = len(stats)
    
    glcm = glcm_skimage(img, distances=distances, angles=angles, levels=levels, kernel_size=kernel_size)

    # glcm shape is (levels, levels, distances, angles, height, width)
    # where height/width are truncated from the smoothing window    
    glcm_h, glcm_w = glcm.shape[4:6]
    
    stat_array = np.zeros((len(distances), len(angles), glcm_h, glcm_w, n_stats))
    
    for stat_i, stat in enumerate(stats):
        for h in range(glcm_h):
            for w in range(glcm_w):
                pass
                stat_array[:,:,h,w,stat_i] = greycoprops(glcm[:,:,:,:,h,w],prop=stat)
            
    # pad the edges back to the original img shape
    trimmed_pixels = int(kernel_size/2)
    edge_pad = (trimmed_pixels,trimmed_pixels)
    no_pad   = (0,0)
    #                   dist     ang     height    width     stat  
    stat_array = np.pad(stat_array, [no_pad, no_pad, edge_pad, edge_pad, no_pad], mode='edge')

    return stat_array


image = np.array([[0, 0, 1, 1, 2, 0, 0],
                  [0, 0, 1, 1, 3, 0, 2],
                  [0, 2, 2, 2, 3, 2, 1],
                  [0, 1, 1, 3, 2, 1, 1],
                  [0, 3, 1, 0, 0, 1, 3],
                  [2, 2, 3, 3, 1, 2, 2]], dtype=np.uint8)

image = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 2, 2, 2, 0, 0],
                  [0, 0, 2, 2, 2, 0, 0],
                  [0, 0, 2, 2, 2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)


glcm1 = glcm_skimage(image, [1], [0,np.pi/2,np.pi, np.pi*1.5], levels=8, kernel_size=3)
glcm2 = fast_glcm(image, [1], [0,np.pi/2, np.pi, np.pi*1.5],levels=8, kernel_size=3)

(glcm1 == glcm2).all()
from timeit import timeit

image = np.random.randint(0,8, size=800*500).reshape((800,500)).astype(np.uint8)

timeit(lambda: glcm_skimage(image, [1], [0], levels=8, kernel_size=3), number=3)
timeit(lambda: fast_glcm(image, [1], [0], levels=8, kernel_size=3), number=3)




