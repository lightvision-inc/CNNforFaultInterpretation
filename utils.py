
import numpy as np

from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows


class FaultsDataset(Dataset):

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return img


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
    padded = np.pad(img, padding, "reflect")
    patches = view_as_windows(padded, (patch_sz, patch_sz), step=step)
    patches = patches.reshape((-1, patch_sz, patch_sz))
    return patches
