from torch.utils.data import Dataset


class SeismicDataset(Dataset):

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return img
