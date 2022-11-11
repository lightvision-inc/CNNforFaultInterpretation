
import scipy
import numpy as np

from itertools import product
from skimage.util.shape import view_as_windows


def normalize_seismic_data(seismic):
    min = seismic.min(axis=(1, 2), keepdims=True)
    max = seismic.max(axis=(1, 2), keepdims=True)
    return (seismic - min) / (max - min)


def get_sliding_wnd_params(img_sz, patch_sz, step):
    overlap_sz = patch_sz - step
    img_width, img_height = img_sz

    horizontal_cnt = int(np.ceil((img_width - overlap_sz) / step))
    new_img_width = step * horizontal_cnt + overlap_sz
    pad_l = int((new_img_width - img_width) / 2)
    pad_r = new_img_width - img_width - pad_l

    vertical_cnt = int(np.ceil((img_height - overlap_sz) / step))
    new_img_height = step * vertical_cnt + overlap_sz
    pad_t = int((new_img_height - img_height) / 2)
    pad_b = new_img_height - img_height - pad_t

    return ((pad_t, pad_b), (pad_l, pad_r)), (horizontal_cnt, vertical_cnt)


def get_sliding_wnd_patches(img, padding, patch_sz, step):
    padded = np.pad(img, padding, 'reflect')
    patches = view_as_windows(padded, (patch_sz, patch_sz), step=step)
    patches = patches.reshape((-1, patch_sz, patch_sz))
    return patches


def recover_img_from_patches(patches, img_sz, padding, overlap_sz):
    assert len(patches.shape) == 4

    i_h, i_w = img_sz
    img = np.zeros(img_sz, dtype=patches.dtype)
    divisor = np.zeros(img_sz, dtype=patches.dtype)

    img = np.pad(img, padding, 'reflect')
    divisor = np.pad(divisor, padding, 'reflect')

    n_h, n_w, p_h, p_w = patches.shape

    s_w = p_w - overlap_sz
    s_h = p_h - overlap_sz

    for y, x in product(range(n_h), range(n_w)):
        patch = patches[y, x]
        img[(y * s_h):(y * s_h) + p_h, (x * s_w):(x * s_w) + p_w] += patch
        divisor[(y * s_h):(y * s_h) + p_h, (x * s_w):(x * s_w) + p_w] += 1

    recovered = img / divisor
    t = padding[0][0]
    l = padding[0][1]
    return recovered[t:t + i_h, l:l + i_w]


cached_2d_windows = dict()

# https://github.com/Vooban/Smoothly-Blend-Image-Patches


def spline_window(wnd_sz, power=2):
    '''
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    '''
    intersection = int(wnd_sz/4)
    wnd_outer = (abs(2*(scipy.signal.triang(wnd_sz))) ** power)/2
    wnd_outer[intersection:-intersection] = 0

    wnd_inner = 1 - (abs(2*(scipy.signal.triang(wnd_sz) - 1)) ** power)/2
    wnd_inner[:intersection] = 0
    wnd_inner[-intersection:] = 0

    wnd = wnd_inner + wnd_outer
    wnd = wnd / np.average(wnd)
    return wnd


def window_2d(wnd_sz, power=2):
    '''
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    '''
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(wnd_sz, power)
    if key in cached_2d_windows:
        wnd = cached_2d_windows[key]
    else:
        wnd = spline_window(wnd_sz, power)
        wnd = np.expand_dims(np.expand_dims(wnd, -1), -1)
        wnd = wnd * wnd.transpose(1, 0, 2)
        cached_2d_windows[key] = wnd
    return wnd
