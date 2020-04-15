"""
    dataset class for all the datasets
"""

from torch.utils.data import Dataset
import numpy as np

DATASETS = {"cifar", "ff", "df", "fs", "f2f"}


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, base_path, mode, compression, dset=None):
    """
    it returns dataset according to its name
    :param name: dataset name
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    else:
        return LatentDataaset(dirpath=base_path + '/' + name + '/', mode=mode, compression=compression, dset=dset)


class LatentDataaset(Dataset):
    """
     Latent Dataset
    """

    def __init__(self, dirpath, mode, compression, dset=None):
        super(Dataset, self).__init__()
        self.dirpath = dirpath
        self.dset = dset
        self.mode = mode
        self.compression = compression

    def __getitem__(self, index):
        # Training images
        if self.mode == 'train':
            if self.dset is None:
                self.data_x = np.load(self.dirpath + '/train_' + self.compression + '_X.npy')
                self.data_y = np.load(self.dirpath + '/train_' + self.compression + '_Y.npy')
            else:
                self.data_x = np.load(self.dirpath + '/train_' + self.compression + '_' + self.dset + '_X.npy')
                self.data_y = np.load(self.dirpath + '/train_' + self.compression + '_' + self.dset + '_Y.npy')
        elif self.mode == 'val':
            if self.dset is None:
                self.data_x = np.load(self.dirpath + '/val_' + self.compression + '_X.npy')
                self.data_y = np.load(self.dirpath + '/val_' + self.compression + '_Y.npy')
            else:
                self.data_x = np.load(self.dirpath + '/val_' + self.compression + '_' + self.dset + '_X.npy')
                self.data_y = np.load(self.dirpath + '/val_' + self.compression + '_' + self.dset + '_Y.npy')

        return self.data_x[index], self.data_y[index]

    def __len__(self):
        if self.mode == 'train':
            if self.dset is None:
                return len(np.load(self.dirpath + '/train_' + self.compression + '_X.npy'))
            else:
                return len(np.load(self.dirpath + '/train_' + self.compression + '_' + self.dset + '_X.npy'))
        elif self.mode == 'val':
            if self.dset is None:
                return len(np.load(self.dirpath + '/val_' + self.compression + '_X.npy'))
            else:
                return len(np.load(self.dirpath + '/val_' + self.compression + '_' + self.dset + '_X.npy'))
