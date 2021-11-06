import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks, view_as_windows
from skimage.filters.rank import entropy
from skimage.measure import shannon_entropy
from skimage.morphology import square
from skimage.feature.texture import greycomatrix, greycoprops

from scipy import ndimage


def fast_glcm(img, nbit=8, kernel_size=5):
    # https://github.com/tzm030329/GLCM/blob/master/fast_glcm.py
    assert img.dtype == np.uint8
    #mi, ma = vmin, vmax
    h,w = img.shape

    # digitize
    gl1 = img
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1
            
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    kernel[:,-1] = 0
    glcm  = glcm.copy()
    for i in range(nbit):
        for j in range(nbit):
            # correlate is the moving window sum, based on kernel weights
            glcm[i,j] = ndimage.correlate(glcm[i,j], kernel, mode='nearest')

    #glcm = glcm.astype(np.float32) / glcm.sum((0,1))
    glcm = glcm.astype(np.float32)
    return glcm

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

def glcm_contrast(m, levels=8):
    glcm = greycomatrix(m, [1],[0], levels=levels, symmetric=False, normed=False)
    return greycoprops(glcm, 'contrast')


img = np.arange(100).reshape((10,10)).astype(np.uint8)
img 


glcm = greycomatrix(img, [1],[0, np.pi/2] )



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

as_win = view_as_windows(image, window_shape=3, step=1)
glcm1 = np.zeros((8,8)+as_win.shape[0:2])
for i in range(as_win.shape[0]):
    for ii in range(as_win.shape[1]):
        glcm1[:,:,i,ii] = greycomatrix(as_win[i,ii], [1],[0], levels=8, symmetric=False, normed=False)[:,:,0,0]
        #glcm1[i,ii] = glcm_contrast(as_win[i,ii]).ravel().round(3)


glcm2 = fast_glcm(image, kernel_size=3)

contrast2 = fast_glcm_contrast(image).round(3)















