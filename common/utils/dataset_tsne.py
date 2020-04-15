"""
    dataset class for all the datasets
"""
import PIL.ImageOps
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import transforms
import numpy as np
import cv2

DATASETS = {"ff", "f2f", "orig", "df", "fs"}


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, base_path, transform=None):
    """
    it returns dataset according to its name
    :param name: dataset name
    :param base_path: base path
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    else:
        return CustomData(name=name, base_path=base_path, transform=transform)


class CustomData(Dataset):
    """
    CustomData dataset
    """

    def __init__(self, name, base_path, transform=None, should_invert=False):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.inputFolderDataset = dset.ImageFolder(root=self.base_path + '/')
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images
        input_images = self.inputFolderDataset.imgs
        # Assign label to class
        input_images = [(t[0], 0) if "orig" in t[0] else (t[0], 1) if "f2f" in t[0] else (t[0], 2) if "df" in t[0] else (t[0], 3) for t in
                        input_images]
        input_img = cv2.imread(input_images[index][0])
        input_img = np.array(input_img, dtype='uint8')
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        if self.should_invert:
            input_img = PIL.ImageOps.invert(input_img)

        if self.transform is not None:
            input_img_as_tensor = self.transform(input_img)
        else:
            input_img_as_tensor = self.to_tensor(input_img)

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.inputFolderDataset.imgs)
